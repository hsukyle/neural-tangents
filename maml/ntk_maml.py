import jax.numpy as np
from jax import random
from jax import jit, grad, vmap

from functools import partial
from network import mlp
from data import sinusoid_task, taskbatch
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sinusoid',
                    help='sinusoid or omniglot or miniimagenet')
parser.add_argument('--n_hidden_layer', type=int, default=2)
parser.add_argument('--n_hidden_unit', type=int, default=49)
parser.add_argument('--bias_coef', type=float, default=1.0)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--outer_step_size', type=float, default=1e-3)
parser.add_argument('--outer_opt_alg', type=str, default='adam',
                    help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='sgd',
                    help='sgd or momentum or adam')
parser.add_argument('--inner_step_size', type=float, default=5e-1)
parser.add_argument('--n_inner_step', type=int, default=3)
parser.add_argument('--task_batch_size', type=int, default=8)
parser.add_argument('--n_train_task', type=int, default=80000)
parser.add_argument('--n_support', type=int, default=20)
parser.add_argument('--n_query', type=int, default=20)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='exp006')
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
net_init, f = mlp(n_output=1,
                  n_hidden_layer=args.n_hidden_layer,
                  n_hidden_unit=args.n_hidden_unit,
                  bias_coef=args.bias_coef,
                  activation=args.activation,
                  norm=args.norm)

# loss functions
loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
grad_loss = jit(grad(lambda p, x, y: loss(f(p, x), y)))
param_loss = jit(lambda p, x, y: loss(f(p, x), y))

# optimizers    #TODO: separate optimizers for nonlinear and linear?
outer_opt_init, outer_opt_update, outer_get_params = select_opt(args.outer_opt_alg, args.outer_step_size)()
inner_opt_init, inner_opt_update, inner_get_params = select_opt(args.inner_opt_alg, args.inner_step_size)()

# consistent task for plotting eval
task_eval = sinusoid_task(n_support=args.n_support, n_query=args.n_query)


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
    fx_train_eval_adapt = f(p, task_eval['x_train'])
    fx_test_eval_adapt = f(p, task_eval['x_test'])
    fx_train_eval_init = f(params_init, task_eval['x_train'])
    fx_test_eval_init = f(params_init, task_eval['x_test'])

    # current task for MAML update
    x_train, y_train, x_test, y_test = task
    p, loss_train, = inner_optimization(
        params_init, x_train, y_train, args.n_inner_step
    )
    loss_test = param_loss(p, x_test, y_test)
    fx_train_adapt = f(p, x_train)
    fx_test_adapt = f(p, x_test)
    fx_train_init = f(params_init, x_train)
    fx_test_init = f(params_init, x_test)

    aux = {}
    for v in [
        'loss_train_eval',
        'loss_test_eval',
        'loss_train',
        'loss_test',
        'fx_train_eval_adapt',
        'fx_test_eval_adapt',
        'fx_train_eval_init',
        'fx_test_eval_init',
        'fx_train_adapt',
        'fx_test_adapt',
        'fx_train_init',
        'fx_test_init',
    ]:
        aux[v] = locals()[v]
    return loss_test, aux


@jit
def maml_loss_batch(params_init, task_batch):
    loss_test_batch, aux_batch = vmap(partial(maml_loss, params_init), in_axes=0)(task_batch)
    aux = {key: np.mean(aux_batch[key]) for key in aux_batch}
    return aux['loss_test'], aux


@jit
def outer_step(i, state, task_batch):
    params = outer_get_params(state)
    g, aux = grad(maml_loss_batch, has_aux=True)(params, task_batch)
    return outer_opt_update(i, g, state), aux


def inner_optimization_lin(params_init, x_train, y_train, n_inner_step):
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
    return p, l_train, f_inner, param_loss_inner


@jit
def maml_loss_lin(params_init, task):
    # evaluation task
    p, loss_train_eval_lin, f_inner, param_loss_inner = inner_optimization_lin(
        params_init, task_eval['x_train'], task_eval['y_train'], args.n_inner_step
    )
    loss_test_eval_lin = param_loss_inner(p, task_eval['x_test'], task_eval['y_test'])
    fx_train_eval_adapt_lin = f_inner(p, task_eval['x_train'])
    fx_test_eval_adapt_lin = f_inner(p, task_eval['x_test'])
    fx_train_eval_init_lin = f_inner(params_init, task_eval['x_train'])
    fx_test_eval_init_lin = f_inner(params_init, task_eval['x_test'])

    # current task for MAML update
    x_train, y_train, x_test, y_test = task
    p, loss_train_lin, f_inner, param_loss_inner, = inner_optimization_lin(
        params_init, x_train, y_train, args.n_inner_step
    )
    loss_test_lin = param_loss_inner(p, x_test, y_test)
    fx_train_adapt_lin = f_inner(p, x_train)
    fx_test_adapt_lin = f_inner(p, x_test)
    fx_train_init_lin = f_inner(params_init, x_train)
    fx_test_init_lin = f_inner(params_init, x_test)

    aux = {}
    for v in [
        'loss_train_eval_lin',
        'loss_test_eval_lin',
        'loss_train_lin',
        'loss_test_lin',
        'fx_train_eval_adapt_lin',
        'fx_test_eval_adapt_lin',
        'fx_train_eval_init_lin',
        'fx_test_eval_init_lin',
        'fx_train_adapt_lin',
        'fx_test_adapt_lin',
        'fx_train_init_lin',
        'fx_test_init_lin',
    ]:
        aux[v] = locals()[v]
    return loss_test_lin, aux


@jit
def maml_loss_lin_batch(params_init, task_batch):
    loss_test_lin_batch, aux_batch = vmap(partial(maml_loss_lin, params_init), in_axes=0)(task_batch)
    aux = {key: np.mean(aux_batch[key]) for key in aux_batch}
    return aux['loss_test_lin'], aux


@jit
def outer_step_lin(i, state, task_batch):
    params = outer_get_params(state)
    g, aux = grad(maml_loss_lin_batch, has_aux=True)(params, task_batch)
    return outer_opt_update(i, g, state), aux


# logging, plotting
win_loss_keys = ['loss_train', 'loss_test', 'loss_train_lin', 'loss_test_lin']
win_loss_eval_keys = ['loss_train_eval', 'loss_test_eval', 'loss_train_eval_lin', 'loss_test_eval_lin']
win_rmse_keys = ['rmse_train_adapt', 'rmse_test_adapt', 'rmse_train_init', 'rmse_test_init']
win_rmse_eval_keys = ['rmse_train_eval_adapt', 'rmse_test_eval_adapt', 'rmse_train_eval_init', 'rmse_test_eval_init']
all_plot_keys = win_loss_keys + win_loss_eval_keys + win_rmse_keys + win_rmse_eval_keys
log = Log(keys=['update'] + all_plot_keys)

_, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 1))
outer_state = outer_opt_init(params)
outer_state_lin = outer_opt_init(params)

rmse = jit(lambda fx, fx_lin: np.sqrt(np.mean(fx - fx_lin) ** 2))

plotter = VisdomPlotter(viz)

for i, task_batch in tqdm(enumerate(taskbatch(task_fn=sinusoid_task,
                                              batch_size=args.task_batch_size,
                                              n_task=args.n_train_task,
                                              n_support=args.n_support,
                                              n_query=args.n_query))):
    outer_state, aux_nonlin = outer_step(
        i=i,
        state=outer_state,
        task_batch=(
            task_batch['x_train'],
            task_batch['y_train'],
            task_batch['x_test'],
            task_batch['y_test']))
    outer_state_lin, aux_lin = outer_step_lin(
        i=i,
        state=outer_state_lin,
        task_batch=(
            task_batch['x_train'],
            task_batch['y_train'],
            task_batch['x_test'],
            task_batch['y_test']))

    aux = {**aux_nonlin, **aux_lin}
    assert (len(aux.keys()) == len(aux_nonlin.keys()) + len(aux_lin.keys()))

    for k in ['train_adapt', 'test_adapt', 'train_init', 'test_init',
              'train_eval_adapt', 'test_eval_adapt', 'train_eval_init', 'test_eval_init']:
        aux[f'rmse_{k}'] = rmse(aux[f'fx_{k}'], aux[f'fx_{k}_lin'])

    log.append([('update', i)])
    log.append([(key, aux[key]) for key in all_plot_keys])

    if (i + 1) % (args.n_train_task // args.task_batch_size // 20) == 0:
        plotter.log_to_line(
            win_name='loss',
            log=log,
            plot_keys=win_loss_keys,
            title='maml sinusoid regression on meta-training tasks',
            xlabel='maml update',
            ylabel='post-adaptation half l2 loss',
            X=log['update']
        )
        plotter.log_to_line(
            win_name='loss_eval',
            log=log,
            plot_keys=win_loss_eval_keys,
            title='maml sinusoid regression on fixed evaluation task',
            xlabel='maml update',
            ylabel='post-adaptation half l2 loss',
            X=log['update']
        )
        plotter.log_to_line(
            win_name='rmse',
            log=log,
            plot_keys=win_rmse_keys,
            title='rmse between nonlinear and linear model outputs on meta-training tasks',
            xlabel='maml update',
            ylabel='rmse',
            X=log['update']
        )
        plotter.log_to_line(
            win_name='rmse_eval',
            log=log,
            plot_keys=win_rmse_eval_keys,
            title='rmse between nonlinear and linear model outputs on fixed evaluation task',
            xlabel='maml update',
            ylabel='rmse',
            X=log['update']
        )

# visualize models

params = outer_get_params(outer_state)
params_lin = outer_get_params(outer_state_lin)
xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
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
plotter.line(win_name='inference_lin', Y=predictions_lin, X=xrange_inputs, name='pre-update predictions', update='append')

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

# serialize
np_dir = os.path.join(args.log_dir, 'np')
os.makedirs(np_dir, exist_ok=True)
onp.save(file=os.path.join(np_dir, f'log'), arr=log)

# serialize visdom envs
viz.save(viz.get_env_list())
