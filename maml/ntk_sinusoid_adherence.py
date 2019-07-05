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
parser.add_argument('--dataset', type=str, default='sinusoid',
                    help='sinusoid or omniglot or miniimagenet')
parser.add_argument('--n_hidden_layer', type=int, default=2)
parser.add_argument('--n_hidden_unit', type=int, default=1024)
parser.add_argument('--bias_coef', type=float, default=0)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default=None)
# parser.add_argument('--outer_step_size', type=float, default=1e-3)
# parser.add_argument('--outer_opt_alg', type=str, default='adam',
# help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='sgd',
                    help='sgd or momentum or adam')
parser.add_argument('--inner_step_size', type=float, default=1e-3)
parser.add_argument('--n_inner_step', type=int, default=30000)
# parser.add_argument('--task_batch_size', type=int, default=8)
# parser.add_argument('--n_train_task', type=int, default=8*20000)
parser.add_argument('--n_support', type=int, default=200)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='ntk-sinusoid-adherence')
# parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f'))
parser.add_argument('--run_name', type=str, default='debug')
parser.add_argument('--n_repeat', type=int, default=3)

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
    net_init, f = mlp(n_output=1,
                      n_hidden_layer=args.n_hidden_layer,
                      n_hidden_unit=args.n_hidden_unit,
                      bias_coef=args.bias_coef,
                      activation=args.activation,
                      norm=args.norm)

    # initialize network
    key = random.PRNGKey(run)
    _, params = net_init(key, (-1, 1))

    # data
    task = sinusoid_task(n_support=args.n_support)
    x_train, y_train, x_test, y_test = task['x_train'], task['y_train'], task['x_test'], task['y_test']

    # linearized network
    f_lin = tangents.linearize(f, params)

    # Create an MSE predictor to solve the NTK equation in function space.
    theta = tangents.ntk(f, batch_size=32)
    g_dd = theta(params, x_train)
    g_td = theta(params, x_test, x_train)
    predictor = tangents.analytic_mse_predictor(g_dd, y_train, g_td)
    import ipdb
    ipdb.set_trace()

    # Get initial values of the network in function space.
    fx_train_ana_init = f(params, x_train)
    fx_test_ana_init = f(params, x_test)

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
    loss = jit(lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2))
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
    grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

    # metric for comparing f and f_lin
    rmse = lambda state, state_lin, x: np.sqrt(np.mean(f(get_params(state), x) - f_lin(get_params(state_lin), x)) ** 2)
    rmse = jit(rmse)

    # log object
    log = Log(['loss_train', 'loss_test', 'loss_train_lin', 'loss_test_lin', 'loss_train_lin_ana', 'loss_test_lin_ana',
               'rmse_train', 'rmse_test', 'iteration'])

    @jit
    def step(i, state):
        params = get_params(state)
        l_train = loss(f(params, x_train), y_train)
        l_test = loss(f(params, x_test), y_test)
        return opt_apply(i, grad_loss(params, x_train, y_train), state), l_train, l_test

    @jit
    def step_lin(i, state):
        params = get_params(state)
        l_train = loss(f_lin(params, x_train), y_train)
        l_test = loss(f_lin(params, x_test), y_test)
        return opt_apply(i, grad_loss_lin(params, x_train, y_train), state), l_train, l_test

    win_loss, win_rmse = None, None
    for i in tqdm(range(args.n_inner_step)):
        rmse_train = rmse(state, state_lin, x_train)
        rmse_test = rmse(state, state_lin, x_test)

        state, l_train, l_test = step(i, state)
        state_lin, l_train_lin, l_test_lin = step_lin(i, state_lin)
        fx_train, fx_test = predictor(fx_train_ana_init, fx_test_ana_init, args.inner_step_size * (i + 1))
        l_train_lin_ana = loss(fx_train, y_train)
        l_test_lin_ana = loss(fx_test, y_test)

        log.append([('iteration', i)])
        log.append([('loss_train', l_train), ('loss_test', l_test), ('loss_train_lin', l_train_lin),
                    ('loss_test_lin', l_test_lin)])
        log.append([('loss_train_lin_ana', l_train_lin_ana), ('loss_test_lin_ana', l_test_lin_ana)])
        log.append([('rmse_train', rmse_train), ('rmse_test', rmse_test)])

        if (i + 1) % (args.n_inner_step // 100) == 0:
            win_loss_keys = ['loss_train', 'loss_test', 'loss_train_lin', 'loss_test_lin', 'loss_train_lin_ana', 'loss_test_lin_ana']
            win_rmse_keys = ['rmse_train', 'rmse_test']
            if win_loss is None:
                win_loss = viz.line(
                    X=log['iteration'],
                    Y=onp.stack([log[key] for key in win_loss_keys], axis=1),
                    opts=dict(title=f'run {run} loss',
                              xlabel='update',
                              ylabel='half l2 loss',
                              legend=['train', 'test', 'train_lin', 'test_lin', 'train_lin_ana', 'test_lin_ana'],
                              dash=onp.array(['dot' for i in range(len(win_loss_keys))]))
                )
            else:
                viz.line(X=log['iteration'],
                         Y=onp.stack([log[key] for key in win_loss_keys], axis=1),
                         win=win_loss, update='replace'
                         )
            if win_rmse is None:
                win_rmse = viz.line(X=log['iteration'],
                                    Y=onp.stack([log[key] for key in win_rmse_keys], axis=1),
                                    opts=dict(title=f'run {run} rmse between f and f_lin',
                                              xlabel='update',
                                              ylabel='rmse',
                                              legend=['train', 'test'],
                                              dash=onp.array(['dot' for i in range(len(win_rmse_keys))])),
                                    )
            else:
                viz.line(X=log['iteration'],
                         Y=onp.stack([log[key] for key in win_rmse_keys], axis=1),
                         win=win_rmse, update='replace'
                         )
            # # scatter plots
            # if win_loss is None:
            #     win_loss = viz.scatter(
            #         X=onp.concatenate([onp.stack([log['iteration'], log[key]], axis=1) for key in win_loss_keys], axis=0),
            #         Y=onp.concatenate([(i+1) * onp.ones(log[key].shape[0]) for i, key in enumerate(win_loss_keys)]),
            #         opts=dict(title=f'run {run} loss',
            #                   xlabel='update',
            #                   ylabel='half l2 loss',
            #                   legend=['train', 'test', 'train_lin', 'test_lin'],
            #                   markersize=7)
            #     )
            # else:
            #     viz.scatter(
            #         X=onp.concatenate([onp.stack([log['iteration'], log[key]], axis=1) for key in win_loss_keys], axis=0),
            #         Y=onp.concatenate([(i + 1) * onp.ones(log[key].shape[0]) for i, key in enumerate(win_loss_keys)]),
            #         win=win_loss, update='replace'
            #     )
            # if win_rmse is None:
            #     win_rmse = viz.scatter(
            #         X=onp.concatenate([onp.stack([log['iteration'], log[key]], axis=1) for key in win_rmse_keys], axis=0),
            #         Y=onp.concatenate([(i+1) * onp.ones(log[key].shape[0]) for i, key in enumerate(win_rmse_keys)]),
            #         opts=dict(title=f'run {run} rmse between f and f_lin',
            #                   xlabel='update',
            #                   ylabel='rmse',
            #                   legend=['train', 'test'],
            #                   markersize=7)
            #     )
            # else:
            #     viz.scatter(
            #         X=onp.concatenate([onp.stack([log['iteration'], log[key]], axis=1) for key in win_rmse_keys],axis=0),
            #         Y=onp.concatenate([(i + 1) * onp.ones(log[key].shape[0]) for i, key in enumerate(win_rmse_keys)]),
            #         win=win_rmse, update='replace'
            #     )

    np_dir = os.path.join(args.log_dir, 'np')
    os.makedirs(np_dir, exist_ok=True)
    onp.save(file=os.path.join(np_dir, f'log_{run}'), arr=log)

    # serialize visdom envs
    # viz.save(viz.get_env_list())