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
from util import Log
from neural_tangents import tangents

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sinusoid', \
    help='sinusoid or omniglot or miniimagenet')
parser.add_argument('--n_hidden_layer', type=int, default=2)
parser.add_argument('--n_hidden_unit', type=int, default=1024)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default=None)
# parser.add_argument('--outer_step_size', type=float, default=1e-3)
# parser.add_argument('--outer_opt_alg', type=str, default='adam', \
    # help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='momentum', \
    help='sgd or momentum or adam')
parser.add_argument('--inner_step_size', type=float, default=1e-2)
parser.add_argument('--n_inner_step', type=int, default=1000)
# parser.add_argument('--task_batch_size', type=int, default=8)
# parser.add_argument('--n_train_task', type=int, default=8*20000)
parser.add_argument('--n_support', type=int, default=100)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='ntk-sinusoid')
parser.add_argument('--run_name', type=str, default= datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f'))
parser.add_argument('--n_repeat', type=int, default=10)

# derive additional args and serialize
args = parser.parse_args()
args.log_dir = os.path.join(args.output_dir, args.exp_name, args.run_name)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(obj=vars(args), fp=open(os.path.join(args.log_dir, 'config.json'), 'w'), sort_keys=True, indent=4)

# initialize visdom
viz = Visdom(port=8000, env=f'{args.exp_name}_{args.run_name}')
viz.text(json.dumps(obj=vars(args), sort_keys=True, indent=4))

for run in tqdm(range(args.n_repeat)):
    # build network
    net_init, f = mlp(n_output=1, n_hidden_layer=args.n_hidden_layer, n_hidden_unit=args.n_hidden_unit, activation=args.activation, norm=args.norm)

    # initialize network
    key = random.PRNGKey(42)
    _, params = net_init(key, (-1, 1))

    # data
    task = sinusoid_task(n_support=args.n_support)
    x_train, y_train, x_test, y_test = task['x_train'], task['y_train'], task['x_test'], task['y_test']

    # linearized network
    f_lin = tangents.linearize(f, params)

    # optimizer for f and f_lin
    if args.inner_opt_alg == 'sgd':
        optimizer = partial(optimizers.sgd, step_size=args.inner_step_size)
    elif args.inner_opt_alg == 'momentum':
        optimizer = partial(optimizers.momentum, step_size=args.inner_step_size, mass=0.9)
    elif args.inner_opt_alg == 'adam':
        optimizer = partial(optimizers.adam, step_size=args.inner_step_size)
    else:
        raise ValueError

    opt_init, opt_apply, get_params = optimizer()
    opt_apply = jit(opt_apply)
    state = opt_init(params)
    state_lin = opt_init(params)

    # loss function
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    loss = jit(loss)
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
    grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

    # metric for comparing f and f_lin
    rmse = lambda state, state_lin, x: np.sqrt(np.mean(f(get_params(state), x) - f_lin(get_params(state_lin), x)) ** 2)
    rmse = jit(rmse)

    # log object
    log = Log(['loss_train', 'loss_test', 'loss_train_lin', 'loss_test_lin', 'rmse_train', 'rmse_test', 'iteration'])

    def step(i, state, f, grad_loss):
        params = get_params(state)
        l_train = loss(f(params, x_train), y_train)
        l_test = loss(f(params, x_test), y_test)
        return opt_apply(i, grad_loss(params, x_train, y_train), state), l_train, l_test

    win_loss, win_rmse = None, None
    for i in tqdm(range(args.n_inner_step)):
        # params = get_params(state)
        # loss_np = onp.append(loss_np, onp.array(loss(f(params, x_train), y_train)))
        # state = opt_apply(i, grad_loss(params, x, y), state)

        # params_lin = get_params(state_lin)
        # loss_lin_np = onp.append(loss_lin_np, onp.array(loss(f_lin(params_lin,))))
        # state_lin = opt_apply(i, grad_loss_lin(params_lin, x, y), state_lin)
        rmse_train = rmse(state, state_lin, x_train)
        rmse_test = rmse(state, state_lin, x_test)

        state, l_train, l_test = step(i, state, f, grad_loss)
        state_lin, l_train_lin, l_test_lin = step(i, state_lin, f_lin, grad_loss_lin)

        log.append([('iteration', i)])
        log.append([('loss_train', l_train), ('loss_test', l_test), ('loss_train_lin', l_train_lin), ('loss_test_lin', l_test_lin)])
        log.append([('rmse_train', rmse_train), ('rmse_test', rmse_test)])

        if (i + 1) % (args.n_inner_step // 100) == 0:
            if win_loss is None:
                win_loss = viz.line(X=log['iteration'], Y=onp.stack([log['loss_train'], log['loss_test'], log['loss_train_lin'], log['loss_test_lin']], axis=1), \
                    opts=dict(title='loss', xlabel='update', ylabel='half l2 loss', legend=['train', 'test', 'train_lin', 'test_lin'])
                )
            else:
                viz.line(X=log['iteration'], Y=onp.stack([log['loss_train'], log['loss_test'], log['loss_train_lin'], log['loss_test_lin']], axis=1), \
                    win=win_loss, update='replace'
                )
            if win_rmse is None:
                win_rmse = viz.line(X=log['iteration'], Y=onp.stack([log['rmse_train'], log['rmse_test']], axis=1), \
                    opts=dict(title='rmse between f and f_lin', xlabel='update', ylabel='rmse', legend=['train', 'test'])
                )
            else:
                viz.line(X=log['iteration'], Y=onp.stack([log['rmse_train'], log['rmse_test']], axis=1), \
                    win=win_rmse, update='replace'
                )

        np_dir = os.path.join(args.log_dir, 'np')
        os.makedirs(np_dir, exist_ok=True)
        onp.save(file=os.path.join(np_dir, f'log_{run}'), arr=log)
