import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.tree_util import tree_multimap
from jax import jit, grad, vmap

from functools import partial
from network import mlp
from data import sinusoid_task, taskbatch
import os
import matplotlib.pyplot as plt
import plotly
import numpy as onp
import argparse
import ipdb
from tqdm import tqdm
import datetime
import json
from visdom import Visdom
from util import Log, select_opt
from neural_tangents import tangents

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sinusoid',
                    help='sinusoid or omniglot or miniimagenet')
parser.add_argument('--n_hidden_layer', type=int, default=2)
parser.add_argument('--n_hidden_unit', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--outer_step_size', type=float, default=1e-3)
parser.add_argument('--outer_opt_alg', type=str, default='adam',
                    help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='sgd',
                    help='sgd or momentum or adam')
parser.add_argument('--inner_step_size', type=float, default=5e-2)
parser.add_argument('--n_inner_step', type=int, default=2)
parser.add_argument('--task_batch_size', type=int, default=1)
parser.add_argument('--n_train_task', type=int, default=20000)
parser.add_argument('--n_support', type=int, default=20)
parser.add_argument('--n_query', type=int, default=50)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='exp003')
# parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f'))
parser.add_argument('--run_name', type=str, default='debug')

# derive additional args and serialize
args = parser.parse_args()
assert args.task_batch_size == 1
args.log_dir = os.path.join(args.output_dir, args.exp_name, args.run_name)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(obj=vars(args), fp=open(os.path.join(args.log_dir, 'config.json'), 'w'), sort_keys=True, indent=4)

# initialize visdom
viz = Visdom(port=8000, env=f'{args.exp_name}_{args.run_name}')
viz.text(json.dumps(obj=vars(args), sort_keys=True, indent=4))

# build network
net_init, f = mlp(n_output=1, n_hidden_layer=args.n_hidden_layer, n_hidden_unit=args.n_hidden_unit,
                  activation=args.activation, norm=args.norm)

# loss functions
loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
grad_loss = jit(grad(lambda p, x, y: loss(f(p, x), y)))
param_loss = jit(lambda p, x, y: loss(f(p, x), y))

# optimizers
outer_opt_init, outer_opt_update, outer_get_params = select_opt(args.outer_opt_alg, args.outer_step_size)()
inner_opt_init, inner_opt_update, inner_get_params = select_opt(args.inner_opt_alg, args.inner_step_size)()

# consistent task for plotting eval
task_eval = sinusoid_task(n_support=args.n_support, n_query=args.n_query)


def inner_optimization(params_init, x_train, y_train, n_inner_step):
    f_inner = tangents.linearize(f, params_init)
    grad_loss_inner = jit(grad(lambda p, x, y: loss(f_inner(p, x), y)))
    param_loss_inner = jit(lambda p, x, y: loss(f_inner(p, x), y))

    state = inner_opt_init(params_init)
    for i in range(n_inner_step):
        p = inner_get_params(state)
        g = grad_loss_inner(p, x_train, y_train)
        state = inner_opt_update(i, g, state)
    p = inner_get_params(state)
    l_train = param_loss_inner(p, x_train, y_train)
    return p, l_train, param_loss_inner


@jit
def maml_loss(params_init, task):
    p, l_train_eval, param_loss_inner = inner_optimization(params_init, task_eval['x_train'], task_eval['y_train'],
                                                            args.n_inner_step)
    l_test_eval = param_loss_inner(p, task_eval['x_test'], task_eval['y_test'])

    # current task for MAML update
    x_train, y_train, x_test, y_test = task
    p, l_train, param_loss_inner = inner_optimization(params_init, x_train, y_train, args.n_inner_step)
    l_test = param_loss_inner(p, x_test, y_test)
    aux = dict(loss_train=l_train, loss_test=l_test, loss_train_eval=l_train_eval, loss_test_eval=l_test_eval)
    return l_test, aux


@jit
def outer_step(i, state, task):
    params = outer_get_params(state)
    # g, aux = grad(maml_loss, has_aux=True)(params, task)
    g, aux = grad(maml_loss, has_aux=True)(params, task)
    return outer_opt_update(i, g, state), aux


# logging, plotting
aux_keys = ['loss_train', 'loss_test']
aux_eval_keys = ['loss_train_eval', 'loss_test_eval']
log = Log(keys=['update'] + aux_keys + aux_eval_keys)
win_loss, win_loss_eval = None, None

_, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 1))
state = outer_opt_init(params)
for i, task_batch in tqdm(enumerate(taskbatch(task_fn=sinusoid_task,
                                              batch_size=args.task_batch_size,
                                              n_task=args.n_train_task,
                                              n_support=args.n_support,
                                              n_query=args.n_query))):
    state, aux = outer_step(i=i, state=state, task=(task_batch['x_train'],
                                                    task_batch['y_train'],
                                                    task_batch['x_test'],
                                                    task_batch['y_test']))
    log.append([(key, aux[key]) for key in aux_keys + aux_eval_keys] + [('update', i)])

    if (i + 1) % (args.n_train_task // args.task_batch_size // 100) == 0:
        win_loss_X = log['update']
        win_loss_Y = onp.stack([log[key] for key in aux_keys] +
                               [onp.convolve(log[key], [0.05] * 20, 'same') for key in aux_keys], axis=1)
        if win_loss is None:
            win_loss = viz.line(
                X=win_loss_X,
                Y=win_loss_Y,
                opts=dict(title=f'maml sinusoid regression on meta-training tasks',
                          xlabel='update',
                          ylabel='half l2 loss',
                          legend=['train', 'test', 'train_smooth', 'test_smooth'],
                          dash=onp.array(['dot' for i in range(len(aux_keys) * 2)]))
            )
        else:
            viz.line(
                X=win_loss_X,
                Y=win_loss_Y,
                win=win_loss,
                update='replace'
            )

        win_loss_eval_X = log['update']
        win_loss_eval_Y = onp.stack([log[key] for key in aux_eval_keys] +
                                    [onp.convolve(log[key], [0.05] * 20, 'same') for key in aux_eval_keys], axis=1)
        if win_loss_eval is None:
            win_loss_eval = viz.line(
                X=win_loss_eval_X,
                Y=win_loss_eval_Y,
                opts=dict(title=f'maml sinusoid regression on evaluation task',
                          xlabel='update',
                          ylabel='half l2 loss',
                          legend=['train', 'test', 'train_smooth', 'test_smooth'],
                          dash=onp.array(['dot' for i in range(len(aux_eval_keys) * 2)]))
            )
        else:
            viz.line(
                X=win_loss_eval_X,
                Y=win_loss_eval_Y,
                win=win_loss_eval,
                update='replace'
            )

# visualize model
params_maml = outer_get_params(state)
xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
targets = np.sin(xrange_inputs)
win_inference = viz.line(
    Y=targets,
    X=xrange_inputs,
    name='target',
    opts=dict(title='sinusoid inference',
              xlabel='x',
              ylabel='y',
              )
)

f_lin = tangents.linearize(f, params_maml)
grad_loss_lin = jit(grad(lambda p, x, y: loss(f_lin(p, x), y)))
param_loss_lin = jit(lambda p, x, y: loss(f_lin(p, x), y))

predictions = f_lin(params_maml, xrange_inputs)
viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='append', name='pre-update predictions')

x1 = onp.random.uniform(low=-5., high=5., size=(args.n_support, 1))
y1 = 1. * onp.sin(x1 + 0.)

state = inner_opt_init(params_maml)
for i in range(1, args.n_inner_step + 1):
    p = inner_get_params(state)
    g = grad_loss_lin(p, x1, y1)
    state = inner_opt_update(i, g, state)
    p = inner_get_params(state)
    predictions = f_lin(p, xrange_inputs)
    viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='append', name=f'{i}-step predictions')

# serialize
np_dir = os.path.join(args.log_dir, 'np')
os.makedirs(np_dir, exist_ok=True)
onp.save(file=os.path.join(np_dir, f'log'), arr=log)

# serialize visdom envs
viz.save(viz.get_env_list())
