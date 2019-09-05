import jax.numpy as np
from jax import random
from jax import jit, grad, vmap
from jax.experimental.stax import logsoftmax, softmax
from jax.tree_util import tree_flatten

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
from scipy import stats

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
parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f'))
parser.add_argument('--noise_std', type=float, default=0.0)
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
    task_eval = sinusoid_task(n_support=args.n_support, n_query=args.n_query, noise_std=args.noise_std)
elif args.dataset == 'omniglot':
    omniglot_splits = load_omniglot(n_support=args.n_support, n_query=args.n_query)
    task_eval = omniglot_task(omniglot_splits['val'], n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'circle':
    task_eval = circle_task(n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
else:
    raise ValueError

#
# # ntk
# ntk = tangents.ntk(f)(params, task_eval['x_train'])
# ipdb.set_trace()

def inner_optimization(params_init, x, y, n_inner_step):
    state = inner_opt_init(params_init)
    for i in range(n_inner_step):
        p = inner_get_params(state)
        g = grad_loss(p, x, y)
        state = inner_opt_update(i, g, state)
    p = inner_get_params(state)
    l = param_loss(p, x, y)
    return p, l

def process_gradient(g_pytree):
    g_list = [x.flatten() for x in tree_flatten(g_pytree)[0]]
    return g_list

@jit
def maml_loss(params_init, task):
    aux = {}

    p, aux['loss_train_eval'] = inner_optimization(
        params_init, task_eval['x_train'], task_eval['y_train'], args.n_inner_step
    )
    aux['loss_test_eval'] = param_loss(p, task_eval['x_test'], task_eval['y_test'])
    aux['fx_train_eval_adapt'] = f(p, task_eval['x_train'])
    aux['fx_test_eval_adapt'] = f(p, task_eval['x_test'])
    aux['fx_train_eval_init'] = f(params_init, task_eval['x_train'])
    aux['fx_test_eval_init'] = f(params_init, task_eval['x_test'])
    if args.dataset in ['omniglot', 'circle']:
        aux['acc_test_eval'] = param_acc(p, task_eval['x_test'], task_eval['y_test'])
        aux['acc_train_eval'] = param_acc(p, task_eval['x_train'], task_eval['y_train'])

    # current task for MAML update
    x_train, y_train, x_test, y_test = task
    p, aux['loss_train'], = inner_optimization(
        params_init, x_train, y_train, args.n_inner_step
    )
    aux['loss_test'] = param_loss(p, x_test, y_test)
    aux['fx_train_adapt'] = f(p, x_train)
    aux['fx_test_adapt'] = f(p, x_test)
    aux['fx_train_init'] = f(params_init, x_train)
    aux['fx_test_init'] = f(params_init, x_test)
    if args.dataset in ['omniglot', 'circle']:
        aux['acc_test'] = param_acc(p, x_test, y_test)
        aux['acc_train'] = param_acc(p, x_train, y_train)

    return aux['loss_test'], aux


@jit
def maml_loss_batch(params_init, task_batch):
    loss_test_batch, aux = vmap(partial(maml_loss, params_init), in_axes=0)(task_batch)
    return np.mean(loss_test_batch), aux


@jit
def outer_step(i, state, task_batch):
    params = outer_get_params(state)
    g, aux = grad(maml_loss_batch, has_aux=True)(params, task_batch)
    return outer_opt_update(i, g, state), aux


def inner_optimization_lin(params_init, x_train, y_train, n_inner_step):
    f_lin = tangents.linearize(f, params_init)
    grad_loss_lin = jit(grad(lambda p, x, y: loss(f_lin(p, x), y)))
    param_loss_lin = jit(lambda p, x, y: loss(f_lin(p, x), y))

    state = inner_opt_init(params_init)
    for i in range(n_inner_step):
        p = inner_get_params(state)
        g = grad_loss_lin(p, x_train, y_train)
        state = inner_opt_update(i, g, state)
    p = inner_get_params(state)
    l_train = param_loss_lin(p, x_train, y_train)
    return p, l_train, f_lin, param_loss_lin


@jit
def maml_loss_lin(params_init, task):
    aux = {}
    # evaluation task
    p, aux['loss_train_eval_lin'], f_lin, param_loss_lin = inner_optimization_lin(
        params_init, task_eval['x_train'], task_eval['y_train'], args.n_inner_step
    )
    aux['loss_test_eval_lin'] = param_loss_lin(p, task_eval['x_test'], task_eval['y_test'])
    aux['fx_train_eval_adapt_lin'] = f_lin(p, task_eval['x_train'])
    aux['fx_test_eval_adapt_lin'] = f_lin(p, task_eval['x_test'])
    aux['fx_train_eval_init_lin'] = f_lin(params_init, task_eval['x_train'])
    aux['fx_test_eval_init_lin'] = f_lin(params_init, task_eval['x_test'])
    if args.dataset in ['omniglot', 'circle']:
        aux['acc_test_eval_lin'] = param_acc(p, task_eval['x_test'], task_eval['y_test'])
        aux['acc_train_eval_lin'] = param_acc(p, task_eval['x_train'], task_eval['y_train'])

    # current task for MAML update
    x_train, y_train, x_test, y_test = task
    p, aux['loss_train_lin'], f_lin, param_loss_lin, = inner_optimization_lin(
        params_init, x_train, y_train, args.n_inner_step
    )
    aux['loss_test_lin'] = param_loss_lin(p, x_test, y_test)
    aux['fx_train_adapt_lin'] = f_lin(p, x_train)
    aux['fx_test_adapt_lin'] = f_lin(p, x_test)
    aux['fx_train_init_lin'] = f_lin(params_init, x_train)
    aux['fx_test_init_lin'] = f_lin(params_init, x_test)
    if args.dataset in ['omniglot', 'circle']:
        aux['acc_test_lin'] = param_acc(p, x_test, y_test)
        aux['acc_train_lin'] = param_acc(p, x_train, y_train)

    return aux['loss_test_lin'], aux


@jit
def maml_loss_lin_batch(params_init, task_batch):
    loss_test_lin_batch, aux = vmap(partial(maml_loss_lin, params_init), in_axes=0)(task_batch)
    return np.mean(loss_test_lin_batch), aux


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
win_tvd_keys = ['tvd_train_adapt', 'tvd_test_adapt', 'tvd_train_init', 'tvd_test_init']
win_tvd_eval_keys = ['tvd_train_eval_adapt', 'tvd_test_eval_adapt', 'tvd_train_eval_init', 'tvd_test_eval_init']
win_rank_eval_keys = ['ntk_train_rank_eval', 'ntk_train_rank_eval_lin']
win_spectrum_eval_keys = [f'ntk_spectrum_{i}_eval' for i in range(args.n_support * args.n_way ** 2)]
win_spectrum_slope_keys = ['ntk_spectrum_log_log_slope']
win_quadratic_keys = ['ntk_label_quadratic', 'ntk_inv_label_quadratic', 'ntk_label_quadratic_normalized']
win_cosine_keys = [f'ntk_evec_{i}_label_cosine' for i in range(args.n_support * args.n_way ** 2)]
win_gradient_keys = [f'g_train_g_test_{i}' for i in range((args.n_hidden_layer + 1) * 2)] + \
    [f'g_train_g_test_normalized_{i}' for i in range((args.n_hidden_layer + 1) * 2)] + \
    ['g_train_g_test', 'g_train_g_test_normalized']
every_iter_plot_keys = win_loss_keys + win_loss_eval_keys + win_rmse_keys + win_rmse_eval_keys + \
    win_tvd_keys + win_tvd_eval_keys + win_gradient_keys
if args.dataset in ['omniglot', 'circle']:
    win_acc_keys = ['acc_train', 'acc_test', 'acc_train_lin', 'acc_test_lin']
    win_acc_eval_keys = ['acc_train_eval', 'acc_test_eval', 'acc_train_eval_lin', 'acc_test_eval_lin']
    every_iter_plot_keys = every_iter_plot_keys + win_acc_keys + win_acc_eval_keys
log = Log(keys=['update'] + every_iter_plot_keys + win_rank_eval_keys + win_spectrum_eval_keys +
               win_spectrum_slope_keys + win_quadratic_keys + win_cosine_keys)

outer_state = outer_opt_init(params)
outer_state_lin = outer_opt_init(params)

rmse = jit(lambda fx, fx_lin: np.sqrt(np.mean(fx - fx_lin) ** 2))
tvd = jit(lambda fx, fx_lin: 0.5 * np.sum(np.abs(softmax(fx) - softmax(fx_lin))))

plotter = VisdomPlotter(viz)

if args.dataset == 'sinusoid':
    task_fn = partial(sinusoid_task, n_support=args.n_support, n_query=args.n_query, noise_std=args.noise_std)
elif args.dataset == 'omniglot':
    task_fn = partial(omniglot_task, split_dict=omniglot_splits['train'], n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
elif args.dataset == 'circle':
    task_fn = partial(circle_task, n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
else:
    raise ValueError

ntk_frequency = 50
plot_update_frequency = 100
for i, task_batch in tqdm(enumerate(taskbatch(task_fn=task_fn,
                                              batch_size=args.task_batch_size,
                                              n_task=args.n_train_task)),
                          total=args.n_train_task // args.task_batch_size):
    aux = dict()
    # ntk
    if i == 0 or (i + 1) % (args.n_train_task // args.task_batch_size // ntk_frequency) == 0:
        ntk = tangents.ntk(f, batch_size=100)(outer_get_params(outer_state), task_eval['x_train'])
        aux['ntk_train_rank_eval'] = onp.linalg.matrix_rank(ntk)
        f_lin = tangents.linearize(f, outer_get_params(outer_state_lin))
        ntk_lin = tangents.ntk(f_lin, batch_size=100)(outer_get_params(outer_state_lin), task_eval['x_train'])
        aux['ntk_train_rank_eval_lin'] = onp.linalg.matrix_rank(ntk_lin)
        log.append([(key, aux[key]) for key in win_rank_eval_keys])

        # spectrum
        evals, evecs = onp.linalg.eigh(ntk)     # eigenvectors are columns
        for j in range(len(evals)):
            aux[f'ntk_spectrum_{j}_eval'] = evals[j]
        log.append([(key, aux[key]) for key in win_spectrum_eval_keys])

        evals = evals.clip(min=1e-10)
        ind = onp.arange(len(evals)) + 1  # +1 because we are taking log
        ind = ind[::-1]
        X = onp.stack([ind, evals], axis=1)
        logX = onp.log10(X)  # don't ignore the clipped eigenvalues when doing linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(logX)
        aux['ntk_spectrum_log_log_slope'] = slope
        log.append([(key, aux[key]) for key in win_spectrum_slope_keys])

        # quadratic
        y = task_eval['y_train'].flatten(order='C')
        ntk_y = onp.matmul(ntk, y)
        quadratic = onp.matmul(ntk_y.T, y)
        aux['ntk_label_quadratic'] = quadratic
        if args.dataset == 'omniglot':
            ntk_inv = onp.linalg.inv(ntk)
            inv_quadratic = y.T @ ntk_inv @ y
            aux['ntk_inv_label_quadratic'] = inv_quadratic
        else:
            aux['ntk_inv_label_quadratic'] = 0
        aux['ntk_label_quadratic_normalized'] = quadratic / onp.linalg.norm(ntk_y) / onp.linalg.norm(y)
        log.append([(key, aux[key]) for key in win_quadratic_keys])

        # cosine
        dot_products = onp.matmul(y.T, evecs)
        evec_norms = onp.linalg.norm(evecs, axis=0)
        cosines = dot_products / (evec_norms * onp.linalg.norm(y))
        for j in range(len(cosines)):
            aux[f'ntk_evec_{j}_label_cosine'] = onp.abs(cosines[j])
        log.append([(key, aux[key]) for key in win_cosine_keys])

    # g_train^T g_test
    params_init = outer_get_params(outer_state)
    g_eval_train_init = process_gradient(grad_loss(params_init, task_eval['x_train'], task_eval['y_train']))
    g_eval_test_init = process_gradient(grad_loss(params_init, task_eval['x_test'], task_eval['y_test']))
    g_train_g_test = [np.dot(a, b) for (a, b) in zip(g_eval_train_init, g_eval_test_init)]
    for j in range(len(g_train_g_test)):
        aux[f'g_train_g_test_{j}'] = g_train_g_test[j]
        aux[f'g_train_g_test_normalized_{j}'] = g_train_g_test[j] / \
                                                np.linalg.norm(g_eval_train_init[j]) / \
                                                np.linalg.norm(g_eval_test_init[j])
    aux['g_train_g_test'] = np.sum(g_train_g_test)
    aux['g_train_g_test_normalized'] = np.sum(g_train_g_test) / \
                                       np.linalg.norm(np.concatenate(g_eval_train_init)) / \
                                       np.linalg.norm(np.concatenate(g_eval_test_init))


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
    assert len(set(aux_nonlin.keys()) & set(aux_lin.keys())) == 0

    aux = {**aux, **aux_nonlin, **aux_lin}

    # rmse and tvd calculation
    for k in ['train_adapt', 'test_adapt', 'train_init', 'test_init',
              'train_eval_adapt', 'test_eval_adapt', 'train_eval_init', 'test_eval_init']:
        aux[f'rmse_{k}'] = vmap(rmse, in_axes=0)(aux[f'fx_{k}'], aux[f'fx_{k}_lin'])
        aux[f'tvd_{k}'] = vmap(tvd, in_axes=0)(aux[f'fx_{k}'], aux[f'fx_{k}_lin'])

    # aggregate
    for (key, value) in aux.items():
        if 'loss' in key or 'acc' in key or 'rmse' in key or 'tvd' in key:
            aux[key] = np.mean(value)

    log.append([('update', i)])
    log.append([(key, aux[key]) for key in every_iter_plot_keys])

    if (i + 1) % (args.n_train_task // args.task_batch_size // plot_update_frequency) == 0:
        if args.dataset == 'sinusoid':
            title = 'maml sinusoid regression'
            loss_ylabel = 'half l2 loss'
        elif args.dataset == 'omniglot':
            title = f'maml omniglot {args.n_way}-way {args.n_support}-shot classification'
            loss_ylabel = 'cross-entropy loss'
        elif args.dataset == 'circle':
            title = f'maml circle {args.n_way}-way {args.n_support}-shot classification'
            loss_ylabel = 'cross-entropy loss'
        else:
            raise ValueError

        plotter.log_to_line(
            win_name='loss',
            log=log,
            plot_keys=win_loss_keys,
            title=f'{title}, meta-training tasks',
            xlabel='maml update',
            ylabel=f'post-adaptation {loss_ylabel}',
            X=log['update'],
            plot_smooth=True
        )
        plotter.log_to_line(
            win_name='loss_eval',
            log=log,
            plot_keys=win_loss_eval_keys,
            title=f'{title}, fixed evaluation task',
            xlabel='maml update',
            ylabel=f'post-adaptation {loss_ylabel}',
            X=log['update'],
            plot_smooth=True
        )
        plotter.log_to_line(
            win_name='rmse',
            log=log,
            plot_keys=win_rmse_keys,
            title='rmse between nonlinear and linear model outputs, meta-training tasks',
            xlabel='maml update',
            ylabel='rmse',
            X=log['update'],
            plot_smooth=True
        )
        plotter.log_to_line(
            win_name='rmse_eval',
            log=log,
            plot_keys=win_rmse_eval_keys,
            title='rmse between nonlinear and linear model outputs, fixed evaluation task',
            xlabel='maml update',
            ylabel='rmse',
            X=log['update'],
            plot_smooth=True
        )
        plotter.log_to_line(
            win_name='tvd',
            log=log,
            plot_keys=win_tvd_keys,
            title='tvd between nonlinear and linear model outputs, meta-training tasks',
            xlabel='maml update',
            ylabel='tvd',
            X=log['update'],
            plot_smooth=True
        )
        plotter.log_to_line(
            win_name='tvd_eval',
            log=log,
            plot_keys=win_tvd_eval_keys,
            title='tvd between nonlinear and linear model outputs, fixed evaluation task',
            xlabel='maml update',
            ylabel='tvd',
            X=log['update'],
            plot_smooth=True
        )
        plotter.log_to_line(
            win_name='g_train_g_test',
            log=log,
            plot_keys=win_gradient_keys,
            title='alignment between gradients using train and test data, fixed evaluation task',
            xlabel='maml update',
            ylabel='dot product',
            X=log['update'],
            plot_smooth=False
        )
        if args.dataset in ['omniglot', 'circle']:
            plotter.log_to_line(
                win_name='acc',
                log=log,
                plot_keys=win_acc_keys,
                title='classification accuracy, meta-training tasks',
                xlabel='maml update',
                ylabel='post-adaptation classification accuracy',
                X=log['update'],
                plot_smooth=True
            )
            plotter.log_to_line(
                win_name='acc_eval',
                log=log,
                plot_keys=win_acc_eval_keys,
                title='classification accuracy, fixed evaluation task',
                xlabel='maml update',
                ylabel='post-adaptation classification accuracy',
                X=log['update'],
                plot_smooth=True
            )

    if i == 0 or (i + 1) % (args.n_train_task // args.task_batch_size // ntk_frequency) == 0:
        plotter.log_to_line(
            win_name='ntk_rank_eval',
            log=log,
            plot_keys=win_rank_eval_keys,
            title=f'rank of NTK on training data (full: {args.n_support * args.n_way ** 2}), fixed evaluation task',
            xlabel='maml update',
            ylabel='rank',
            X=onp.arange(log['ntk_train_rank_eval'].shape[0]) * (args.n_train_task // args.task_batch_size // ntk_frequency),
            plot_smooth=False
        )
        plotter.log_to_line(
            win_name='ntk_spectrum_eval',
            log=log,
            plot_keys=win_spectrum_eval_keys,
            title=f'spectrum of NTK on training data, fixed evaluation task',
            xlabel='maml update',
            ylabel='eigenvalue',
            X=onp.arange(log['ntk_spectrum_0_eval'].shape[0]) * (args.n_train_task // args.task_batch_size // ntk_frequency),
            plot_smooth=False
        )
        plotter.log_to_line(
            win_name='ntk_spectrum_log_log_slope',
            log=log,
            plot_keys=win_spectrum_slope_keys,
            title=f'slope of log-log plot of NTK spectrum, fixed evaluation task, support data',
            xlabel='maml update',
            ylabel='slope',
            X=onp.arange(log['ntk_spectrum_log_log_slope'].shape[0]) * (args.n_train_task // args.task_batch_size // ntk_frequency),
            plot_smooth=False
        )
        plotter.log_to_line(
            win_name='ntk_label_quadratic',
            log=log,
            plot_keys=win_quadratic_keys,
            title=f'quadratic of labels and ntk, fixed evaluation task, support data',
            xlabel='maml update',
            ylabel='y^T ntk y',
            X=onp.arange(log['ntk_label_quadratic'].shape[0]) * (args.n_train_task // args.task_batch_size // ntk_frequency),
            plot_smooth=False
        )
        plotter.log_to_line(
            win_name='ntk_evec_label_cosine',
            log=log,
            plot_keys=win_cosine_keys,
            title=f'alignment of labels and ntk, fixed evaluation task, support data',
            xlabel='maml update',
            ylabel='absolute cosine similarity',
            X=onp.arange(log['ntk_evec_0_label_cosine'].shape[0]) * (args.n_train_task // args.task_batch_size // ntk_frequency),
            plot_smooth=False
        )

        # fig, (axis1, axis2) = plt.subplots(2, 1)
        # axis1.matshow(onp.asarray(ntk))
        # axis1.set_title(f'ntk of f on x_train, iter {i}, fixed evaluation task')
        # axis2.matshow(onp.asarray(ntk_lin))
        # axis2.set_title(f'ntk of f_lin on x_train, iter {i}, fixed evaluation task')
        # viz.matplot(fig, opts=dict(width=800, height=600))
        # plt.close('all')

        # ntk spectrum log-log plot
        evals, _ = onp.linalg.eigh(ntk)
        evals = evals.clip(min=1e-10)   # numerical errors can result in small negative values, which shouldn't happen
        ind = onp.arange(len(evals)) + 1    # +1 because we are taking log
        ind = ind[::-1]
        X = onp.stack([ind, evals], axis=1)
        logX = onp.log10(X)     # don't ignore the clipped eigenvalues when doing linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(logX)

        if args.dataset == 'sinusoid':
            ytickmin = -10
        elif args.dataset == 'omniglot':
            ytickmin = -4

        spectrum_plot = viz.scatter(
            X=X, name='raw',
            opts=dict(title=f'NTK spectrum, iter {i}, training data, fixed evaluation task', xtype='log', ytype='log',
                      xlabel='index', ylabel='eigenvalue', width=800, height=600,
                      ytickmin=ytickmin, ytickmax=3, showlegend=True)
        )

        X_fit = onp.stack([ind, 10**(slope * onp.log10(ind) + intercept)], axis=1)
        viz.line(Y=X_fit[:, 1], X=X_fit[:, 0], win=spectrum_plot, update='append',
                 name=f'slope {slope:.3f}, intercept {intercept:.3f}')


# visualize models
if args.dataset == 'sinusoid':
    params = outer_get_params(outer_state)
    params_lin = outer_get_params(outer_state_lin)
    xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
    targets = np.sin(xrange_inputs) + onp.random.normal(scale=args.noise_std, size=xrange_inputs.shape)
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

# serialize
np_dir = os.path.join(args.log_dir, 'np')
os.makedirs(np_dir, exist_ok=True)
onp.save(file=os.path.join(np_dir, f'log'), arr=log)

# serialize visdom envs
viz.save(viz.get_env_list())

