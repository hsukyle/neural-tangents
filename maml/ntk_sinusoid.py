import jax.numpy as np
from jax import random
from jax import jit, grad, vmap
from jax.experimental.stax import logsoftmax, softmax

from functools import partial
from network import mlp, conv_net
from data import sinusoid_task, taskbatch, omniglot_task, load_omniglot, circle_task
import os
import numpy as onp
import argparse
import ipdb
from tqdm import tqdm
import datetime
import json
from visdom import Visdom
from util import Log, select_opt, VisdomPlotter
from neural_tangents import tangents
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='omniglot',
                    help='sinusoid or circle or omniglot')
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
parser.add_argument('--inner_step_size', type=float, default=5e0)
parser.add_argument('--n_inner_step', type=int, default=1)
parser.add_argument('--task_batch_size', type=int, default=16)
parser.add_argument('--n_train_task', type=int, default=16 * 10000)
parser.add_argument('--n_way', type=int, default=5)
parser.add_argument('--n_support', type=int, default=3)
parser.add_argument('--n_query', type=int, default=15)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='exp010')
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

elif args.dataset == 'circle':
    net_init, f = mlp(n_output=args.n_way,
                      n_hidden_layer=args.n_hidden_layer,
                      n_hidden_unit=args.n_hidden_unit,
                      bias_coef=args.bias_coef,
                      activation=args.activation,
                      norm=args.norm)
    _, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 2))

else:
    raise ValueError

# loss functions
if args.dataset == 'sinusoid':
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
elif args.dataset in ['omniglot', 'circle']:
    loss = lambda fx, targets: -np.sum(logsoftmax(fx) * targets) / targets.shape[0]
    acc = lambda fx, targets: np.mean(np.argmax(logsoftmax(fx), axis=-1) == np.argmax(targets, axis=-1))
    param_acc = jit(lambda p, x, y: acc(f(p, x), y))
else:
    raise ValueError

grad_loss = jit(grad(lambda p, x, y: loss(f(p, x), y)))
param_loss = jit(lambda p, x, y: loss(f(p, x), y))

# optimizers    #TODO: separate optimizers for nonlinear and linear?
outer_opt_init, outer_opt_update, outer_get_params = select_opt(args.outer_opt_alg, args.outer_step_size)()
inner_opt_init, inner_opt_update, inner_get_params = select_opt(args.inner_opt_alg, args.inner_step_size)()

# consistent task for plotting eval
if args.dataset == 'sinusoid':
    task_eval = sinusoid_task(n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'omniglot':
    omniglot_splits = load_omniglot(n_support=args.n_support, n_query=args.n_query)
    task_eval = omniglot_task(omniglot_splits['val'], n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'circle':
    task_eval = circle_task(n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
else:
    raise ValueError

outer_state = outer_opt_init(params)
outer_state_lin = outer_opt_init(params)

plotter = VisdomPlotter(viz)

if args.dataset == 'sinusoid':
    task_fn = partial(sinusoid_task, n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'omniglot':
    task_fn = partial(omniglot_task, split_dict=omniglot_splits['train'], n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'circle':
    task_fn = partial(circle_task, n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
else:
    raise ValueError

# visualize models
if args.dataset == 'sinusoid':
    params = outer_get_params(outer_state)
    params_lin = outer_get_params(outer_state_lin)
    xrange_inputs = np.linspace(-5, 5, 200).reshape(-1, 1)
    targets = np.sin(xrange_inputs)
    plotter.line(
        win_name='inference',
        Y=targets,
        X=xrange_inputs,
        name='target',
        title='nonlinear model sinusoid inference',
        xlabel='x',
        ylabel='y',
    )
    plotter.line(
        win_name='inference_lin',
        Y=targets,
        X=xrange_inputs,
        name='target',
        title='linear model sinusoid inference',
        xlabel='x',
        ylabel='y',
    )

    f_lin = tangents.linearize(f, params_lin)
    grad_loss_lin = jit(grad(lambda p, x, y: loss(f_lin(p, x), y)))
    param_loss_lin = jit(lambda p, x, y: loss(f_lin(p, x), y))

    predictions = f(params, xrange_inputs)
    plotter.line(win_name='inference', Y=predictions, X=xrange_inputs, name='pre-update predictions', update='append')
    predictions_lin = f_lin(params_lin, xrange_inputs)
    plotter.line(win_name='inference_lin', Y=predictions_lin, X=xrange_inputs, name='pre-update predictions',
                 update='append')

    x1 = onp.random.uniform(low=-5., high=5., size=(args.n_support, 1))
    y1 = 1. * onp.sin(x1 + 0.)

    state = inner_opt_init(params)
    state_lin = inner_opt_init(params_lin)
    for i in range(1, args.n_inner_step + 1):
        p = inner_get_params(state)
        g = grad_loss(p, x1, y1)
        state = inner_opt_update(i, g, state)
        p = inner_get_params(state)
        predictions = f(p, xrange_inputs)
        plotter.line(win_name='inference', Y=predictions, X=xrange_inputs, name=f'{i}-step predictions',
                     update='append')

        p_lin = inner_get_params(state_lin)
        g_lin = grad_loss_lin(p_lin, x1, y1)
        state_lin = inner_opt_update(i, g_lin, state_lin)
        p_lin = inner_get_params(state_lin)
        predictions_lin = f_lin(p_lin, xrange_inputs)
        plotter.line(win_name='inference_lin', Y=predictions_lin, X=xrange_inputs, name=f'{i}-step predictions',
                     update='append')

# serialize visdom envs
viz.save(viz.get_env_list())

ipdb.set_trace()
x = 1
