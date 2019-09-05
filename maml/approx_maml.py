import argparse
import os
import json
import datetime
from visdom import Visdom
from tqdm import tqdm
import ipdb

import jax.numpy as np
from jax import random, jit, grad, vmap
from jax.tree_util import tree_flatten, tree_multimap
from jax.lax import stop_gradient

from functools import partial
import numpy as onp

from network import mlp, conv_net
from util import select_opt, Log, VisdomPlotter
from data import sinusoid_task, taskbatch

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='exp020')
parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f'))
parser.add_argument('--debug', action='store_true')

parser.add_argument('--dataset', type=str, default='sinusoid',
                    help='sinusoid or circle or omniglot')
parser.add_argument('--noise_std', type=float, default=0.0)
parser.add_argument('--n_hidden_layer', type=int, default=2)
parser.add_argument('--n_hidden_unit', type=int, default=256)
parser.add_argument('--bias_coef', type=float, default=1.0)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default='none')
parser.add_argument('--opt_alg', type=str, default='adam')
parser.add_argument('--step_size', type=float, default=0.001)
parser.add_argument('--task_batch_size', type=int, default=16)
parser.add_argument('--n_train_task', type=int, default=16 * 10000)
parser.add_argument('--n_support', type=int, default=20)
parser.add_argument('--n_query', type=int, default=20)

parser.add_argument('--stop_gradient', action='store_true')
parser.add_argument('--alignment_coefficient', type=float, default=0.005)

# derive additional args and serialize
args = parser.parse_args()
if args.debug:
    args.run_name = 'debug'
args.log_dir = os.path.join(args.output_dir, args.exp_name, args.run_name)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(obj=vars(args), fp=open(os.path.join(args.log_dir, 'config.json'), 'w'), sort_keys=True, indent=4)

# initialize visdom
env_name = f'{args.exp_name}_{args.run_name}'
viz = Visdom(port=8000, env=env_name)
viz.text(json.dumps(obj=vars(args), sort_keys=True, indent=4))

# data
if args.dataset == 'sinusoid':
    task_fn = partial(sinusoid_task, n_support=args.n_support, n_query=args.n_query, noise_std=args.noise_std)

# build network
if args.dataset == 'sinusoid':
    net_init, f = mlp(n_output=1,
                      n_hidden_layer=args.n_hidden_layer,
                      n_hidden_unit=args.n_hidden_unit,
                      bias_coef=args.bias_coef,
                      activation=args.activation,
                      norm=args.norm)
    _, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 1))

# optimizer
opt_init, opt_update, get_params = select_opt(args.opt_alg, args.step_size)()

# loss function
if args.dataset == 'sinusoid':
    def loss(fx, y):
        return 0.5 * np.mean((fx - y) ** 2)

grad_loss = grad(lambda p, x, y: loss(f(p, x), y))
param_loss = lambda p, x, y: loss(f(p, x), y)

def pytree_to_array(pytree):
    return np.concatenate([x.flatten() for x in tree_flatten(pytree)[0]])


# if args.stop_gradient:
#     def loss_alignment(g_support, g_query):
#         return -np.dot(stop_gradient(g_support), g_query)
# else:
#     def loss_alignment(g_support, g_query):
#         return -np.dot(g_support, g_query)


def loss_total(params, task):
    x_support, y_support, x_query, y_query = task
    g_support = pytree_to_array(grad_loss(params, x_support, y_support))
    g_query = pytree_to_array(grad_loss(params, x_query, y_query))

    loss_query = loss(f(params, x_query), y_query)
    # loss_alignment_val = loss_alignment(g_support, g_query)
    if args.stop_gradient:
        g_support = stop_gradient(g_query)
    loss_alignment_val = -np.dot(g_support, g_query)    #TODO normalize
    loss_combined = loss_query + args.alignment_coefficient * loss_alignment_val

    aux = dict()
    aux['loss_query'] = loss_query
    aux['loss_alignment'] = loss_alignment_val
    aux['loss_combined'] = loss_combined
    aux['grad_support_norm'] = np.linalg.norm(g_support)
    aux['grad_query_norm'] = np.linalg.norm(g_query)

    return loss_combined, aux


def loss_batch(params, task_batch):
    loss_b, aux = vmap(partial(loss_total, params), in_axes=0)(task_batch)

    # aggregate batch aux
    for key in list(aux.keys()):
        aux[f'{key}_batch'] = aux[key]
        aux[key] = np.mean(aux[key])

    return np.mean(loss_b), aux


@jit
def step(i, state, task_batch):
    params = get_params(state)
    g, aux = grad(loss_batch, has_aux=True)(params, task_batch)
    return opt_update(i, g, state), aux


task_batch_eval = list(taskbatch(task_fn=task_fn, batch_size=64, n_task=64))[0]


def eval_inner(params, task):
    x_support, y_support, x_query, y_query = task
    grads = grad_loss(params, x_support, y_support)
    # step_size = 1 / np.linalg.norm(pytree_to_array(grads))
    step_size = args.alignment_coefficient
    inner_sgd_fn = lambda g, p: p - step_size * g
    params_updated = tree_multimap(inner_sgd_fn, grads, params)
    loss = param_loss(params_updated, x_query, y_query)

    return loss

@jit
def eval(i, state):
    aux = {}
    params = get_params(state)
    param_array = pytree_to_array(params)
    aux['param_norm_l2'] = np.linalg.norm(param_array, 2)
    aux['param_norm_l1'] = np.linalg.norm(param_array, 1)
    aux['param_norm_l0'] = np.linalg.norm(param_array, 0)
    aux['param_norm_linf'] = np.linalg.norm(param_array, np.inf)
    aux['param_norm_l-inf'] = np.linalg.norm(param_array, -np.inf)

    task_batch = (task_batch_eval['x_train'],
                  task_batch_eval['y_train'],
                  task_batch_eval['x_test'],
                  task_batch_eval['y_test'])
    losses_query = vmap(partial(eval_inner, params), in_axes=0)(task_batch)

    aux['eval_loss_query_finetune_batch'] = losses_query
    aux['eval_loss_query_finetune'] = np.mean(losses_query)

    return aux


state = opt_init(params)

# logging, plotting
win_loss_keys = ['loss_query', 'loss_alignment', 'loss_combined']
win_grad_norm_keys = ['grad_support_norm', 'grad_query_norm']
win_param_norm_keys = ['param_norm_l2', 'param_norm_l1', 'param_norm_l0', 'param_norm_linf', 'param_norm_l-inf']
win_finetune_keys = ['eval_loss_query_finetune']
training_keys = win_loss_keys + win_grad_norm_keys
eval_keys = win_param_norm_keys + win_finetune_keys
log = Log(keys=['update'] + training_keys + eval_keys)
plotter = VisdomPlotter(viz)

plot_update_period = 100
eval_period = 1
total_iters = args.n_train_task // args.task_batch_size
for i, task_batch in tqdm(enumerate(taskbatch(task_fn=task_fn,
                                              batch_size=args.task_batch_size,
                                              n_task=args.n_train_task)),
                          total=total_iters):
    # optimization step
    state, aux = step(i, state, task_batch=(
        task_batch['x_train'], task_batch['y_train'], task_batch['x_test'], task_batch['y_test']
    ))

    # append iteration and training aux info to log
    log.append([('update', i)])
    log.append([(key, aux[key]) for key in training_keys])

    if i == 0 or (i + 1) % eval_period == 0:
        aux_eval = eval(i, state)
        log.append([(key, aux_eval[key]) for key in eval_keys])

    if (i + 1) % plot_update_period == 0:
        plotter.log_to_line(
            win_name='loss',
            log=log,
            plot_keys=win_loss_keys,
            title='losses, training tasks',
            xlabel='update',
            ylabel='loss',
            X=log['update'],
            plot_smooth=False
        )

        plotter.log_to_line(
            win_name='grad_norm',
            log=log,
            plot_keys=win_grad_norm_keys,
            title='gradient l2 norms, training tasks',
            xlabel='update',
            ylabel='l2 norm',
            X=log['update'],
            plot_smooth=False
        )

        plotter.log_to_line(
            win_name='param_norm',
            log=log,
            plot_keys=win_param_norm_keys,
            title='parameter vector norms',
            xlabel='update',
            ylabel='norm',
            X=onp.arange(log['param_norm_l2'].shape[0]) * eval_period
        )

        plotter.log_to_line(
            win_name='finetune',
            log=log,
            plot_keys=win_finetune_keys,
            title='finetuning on held-out tasks',
            xlabel='update',
            ylabel='loss',
            X=onp.arange(log['eval_loss_query_finetune'].shape[0]) * eval_period
        )

# serialize
np_dir = os.path.join(args.log_dir, 'np')
os.makedirs(np_dir, exist_ok=True)
onp.save(file=os.path.join(np_dir, f'log'), arr=log)

# serialize visdom env
# viz.save(viz.get_env_list())
viz.save([env_name])
