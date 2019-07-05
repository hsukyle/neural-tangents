import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.tree_util import tree_multimap
from jax import jit, grad, vmap
from jax.experimental.stax import logsoftmax

from functools import partial
from network import mlp, conv_net
from data import sinusoid_task, taskbatch, omniglot_task, load_omniglot
import os
import numpy as onp
import argparse
import ipdb
from tqdm import tqdm
import datetime
import json
from visdom import Visdom
from util import Log, select_opt, VisdomPlotter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='omniglot',
                    help='sinusoid or omniglot or miniimagenet')
parser.add_argument('--n_hidden_layer', type=int, default=4)
parser.add_argument('--n_hidden_unit', type=int, default=64)
parser.add_argument('--bias_coef', type=float, default=1.0)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default='batch_norm')
parser.add_argument('--outer_step_size', type=float, default=1e-2)
parser.add_argument('--outer_opt_alg', type=str, default='adam',
                    help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='sgd',
                    help='sgd or momentum or adam')
parser.add_argument('--inner_step_size', type=float, default=5e0/2)
parser.add_argument('--n_inner_step', type=int, default=1*2)
parser.add_argument('--task_batch_size', type=int, default=16)
parser.add_argument('--n_train_task', type=int, default=16*10000)
parser.add_argument('--n_way', type=int, default=5)
parser.add_argument('--n_support', type=int, default=5)
parser.add_argument('--n_query', type=int, default=15)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='exp007')
parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f'))
parser.add_argument('--debug', action='store_true')

# derive additional args and serialize
args = parser.parse_args()
if args.debug:
    args.run_name = 'debug'
args.log_dir = os.path.join(args.output_dir, args.exp_name, args.run_name)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(obj=vars(args), fp=open(os.path.join(args.log_dir, 'config.json'), 'w'), sort_keys=True, indent=4)

# initialize visdom
viz = Visdom(port=8000, env=f'{args.exp_name}_{args.run_name}')
viz.text(json.dumps(obj=vars(args), sort_keys=True, indent=4))

# build network
if args.dataset == 'sinusoid':
    net_init, f = mlp(n_output=1,
                      n_hidden_layer=args.n_hidden_layer,
                      n_hidden_unit=args.n_hidden_unit,
                      bias_coef=args.bias_coef,
                      activation=args.activation,
                      norm=args.norm)
    _, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 1))

elif args.dataset == 'omniglot':
    net_init, f = conv_net(n_output=args.n_way,
                           n_conv_layer=args.n_hidden_layer,
                           n_filter=args.n_hidden_unit,
                           bias_coef=args.bias_coef,
                           activation='relu',
                           norm='None')
    _, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 28, 28, 1))

else:
    raise ValueError

# loss functions
if args.dataset == 'sinusoid':
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
elif args.dataset == 'omniglot':
    loss = lambda fx, targets: -np.sum(logsoftmax(fx) * targets) / targets.shape[0]
    acc = lambda fx, targets: np.mean(np.argmax(logsoftmax(fx), axis=-1) == np.argmax(targets, axis=-1))
    param_acc = jit(lambda p, x, y: acc(f(p, x), y))
else:
    raise ValueError

grad_loss = jit(grad(lambda p, x, y: loss(f(p, x), y)))
param_loss = jit(lambda p, x, y: loss(f(p, x), y))

# optimizers
outer_opt_init, outer_opt_update, outer_get_params = select_opt(args.outer_opt_alg, args.outer_step_size)()
inner_opt_init, inner_opt_update, inner_get_params = select_opt(args.inner_opt_alg, args.inner_step_size)()

# consistent task for plotting eval
if args.dataset == 'sinusoid':
    task_eval = sinusoid_task(n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'omniglot':
    omniglot_splits = load_omniglot(n_support=args.n_support, n_query=args.n_query)
    task_eval = omniglot_task(omniglot_splits['val'], n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)


def inner_optimization(params_init, x_train, y_train, n_inner_step):
    state = inner_opt_init(params_init)
    for i in range(n_inner_step):
        p = inner_get_params(state)
        g = grad_loss(p, x_train, y_train)
        state = inner_opt_update(i, g, state)
    p = inner_get_params(state)
    loss_train = param_loss(p, x_train, y_train)
    return p, loss_train


@jit
def maml_loss(params_init, task):
    p, loss_train_eval = inner_optimization(
        params_init, task_eval['x_train'], task_eval['y_train'], args.n_inner_step
    )
    loss_test_eval = param_loss(p, task_eval['x_test'], task_eval['y_test'])
    if args.dataset == 'omniglot':
        acc_test_eval = param_acc(p, task_eval['x_test'], task_eval['y_test'])
        acc_train_eval = param_acc(p, task_eval['x_train'], task_eval['y_train'])

    # current task for MAML update
    x_train, y_train, x_test, y_test = task
    p, loss_train, = inner_optimization(
        params_init, x_train, y_train, args.n_inner_step
    )
    loss_test = param_loss(p, x_test, y_test)
    if args.dataset == 'omniglot':
        acc_test = param_acc(p, x_test, y_test)
        acc_train = param_acc(p, x_train, y_train)

    aux = {}
    keys = [
        'loss_train_eval',
        'loss_test_eval',
        'loss_train',
        'loss_test',
    ]
    if args.dataset == 'omniglot':
        keys += [
            'acc_train_eval',
            'acc_test_eval',
            'acc_train',
            'acc_test'
        ]
    for key in keys:
        aux[key] = locals()[key]
    return loss_test, aux


@jit
def batch_maml_loss(params_init, task_batch):
    losses_test, aux = vmap(partial(maml_loss, params_init))(task_batch)
    aux = {key: np.mean(aux[key]) for key in aux}
    return aux['loss_test'], aux


@jit
def outer_step(i, state, task_batch):
    params = outer_get_params(state)
    g, aux = grad(batch_maml_loss, has_aux=True)(params, task_batch)
    return outer_opt_update(i, g, state), aux


# logging, plotting
win_loss_keys = ['loss_train', 'loss_test']
win_loss_eval_keys = ['loss_train_eval', 'loss_test_eval']
all_plot_keys = win_loss_keys + win_loss_eval_keys
if args.dataset == 'omniglot':
    win_acc_keys = ['acc_train', 'acc_test']
    win_acc_eval_keys = ['acc_train_eval', 'acc_test_eval']
    all_plot_keys = all_plot_keys + win_acc_keys + win_acc_eval_keys
log = Log(keys=['update'] + all_plot_keys)

outer_state = outer_opt_init(params)

plotter = VisdomPlotter(viz)

if args.dataset == 'sinusoid':
    task_fn = partial(sinusoid_task, n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'omniglot':
    task_fn = partial(omniglot_task, split_dict=omniglot_splits['train'], n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
else:
    raise ValueError

for i, task_batch in tqdm(enumerate(taskbatch(task_fn=task_fn,
                                              batch_size=args.task_batch_size,
                                              n_task=args.n_train_task)),
                          total=args.n_train_task // args.task_batch_size):
    outer_state, aux = outer_step(
        i=i,
        state=outer_state,
        task_batch=(
            task_batch['x_train'],
            task_batch['y_train'],
            task_batch['x_test'],
            task_batch['y_test']))

    log.append([('update', i)])
    log.append([(key, aux[key]) for key in all_plot_keys])

    if (i + 1) % (args.n_train_task // args.task_batch_size // 100) == 0:
        if args.dataset == 'sinusoid':
            title = 'maml sinusoid regression'
            ylabel = 'half l2 loss'
        elif args.dataset == 'omniglot':
            title = f'maml omniglot {args.n_way}-way {args.n_support}-shot classification'
            ylabel = 'cross-entropy loss'
        else:
            raise ValueError

        plotter.log_to_line(
            win_name='loss',
            log=log,
            plot_keys=win_loss_keys,
            title=f'{title}, meta-training tasks',
            xlabel='maml update',
            ylabel=f'post-adaptation {ylabel}',
            X=log['update']
        )
        plotter.log_to_line(
            win_name='loss_eval',
            log=log,
            plot_keys=win_loss_eval_keys,
            title=f'{title}, fixed evaluation task',
            xlabel='maml update',
            ylabel=f'post-adaptation {ylabel}',
            X=log['update']
        )

        if args.dataset == 'omniglot':
            plotter.log_to_line(
                win_name='acc',
                log=log,
                plot_keys=win_acc_keys,
                title='classification accuracy, meta-training tasks',
                xlabel='maml update',
                ylabel='post-adaptation classification accuracy',
                X=log['update']
            )
            plotter.log_to_line(
                win_name='acc_eval',
                log=log,
                plot_keys=win_acc_eval_keys,
                title='classification accuracy, fixed evaluation task',
                xlabel='maml update',
                ylabel='post-adaptation classification accuracy',
                X=log['update']
            )

# visualize models
if args.dataset == 'sinusoid':
    params = outer_get_params(outer_state)
    xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
    targets = np.sin(xrange_inputs)
    win_inference = viz.line(
        Y=targets,
        X=xrange_inputs,
        name='target',
        opts=dict(title='nonlinear model sinusoid inference',
                  xlabel='x',
                  ylabel='y',
                  )
    )

    predictions = f(params, xrange_inputs)
    viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='append', name='pre-update predictions')

    x1 = onp.random.uniform(low=-5., high=5., size=(args.n_support, 1))
    y1 = 1. * onp.sin(x1 + 0.)

    state = inner_opt_init(params)
    for i in range(1, args.n_inner_step + 1):
        p = inner_get_params(state)
        g = grad_loss(p, x1, y1)
        state = inner_opt_update(i, g, state)
        p = inner_get_params(state)
        predictions = f(p, xrange_inputs)
        viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='append', name=f'{i}-step predictions')

# serialize
np_dir = os.path.join(args.log_dir, 'np')
os.makedirs(np_dir, exist_ok=True)
onp.save(file=os.path.join(np_dir, f'log'), arr=log)

# serialize visdom envs
viz.save(viz.get_env_list())
