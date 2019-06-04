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
parser.add_argument('--n_hidden_unit', type=int, default=40)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--outer_step_size', type=float, default=1e-3)
parser.add_argument('--outer_opt_alg', type=str, default='adam',
                    help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='sgd',
                    help='sgd or momentum or adam')
parser.add_argument('--inner_step_size', type=float, default=1e-2)
parser.add_argument('--n_inner_step', type=int, default=10)
parser.add_argument('--task_batch_size', type=int, default=1)
parser.add_argument('--n_train_task', type=int, default=10000)
parser.add_argument('--n_support', type=int, default=10)
parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
parser.add_argument('--exp_name', type=str, default='ntk-maml-adherence')
parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f'))
# parser.add_argument('--run_name', type=str, default='debug')

# derive additional args and serialize
args = parser.parse_args()
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
grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
param_loss = jit(lambda params, x, y: loss(f(params, x), y))

# optimizers
outer_opt_init, outer_opt_update, outer_get_params = select_opt(args.outer_opt_alg, args.outer_step_size)()
inner_opt_init, inner_opt_update, inner_get_params = select_opt(args.inner_opt_alg, args.inner_step_size)()


def inner_optimization(params, x, y, n_inner_step):
    state = inner_opt_init(params)
    for i in range(n_inner_step):
        params = inner_get_params(state)
        g = grad_loss(params, x, y)
        state = inner_opt_update(i, g, state)
    params = inner_get_params(state)
    l = param_loss(params, x, y)
    return params, l


@jit
def maml_loss(params, task):
    x_train, y_train, x_test, y_test = task
    params, l_train = inner_optimization(params, x_train, y_train, args.n_inner_step)
    l_test = param_loss(params, x_test, y_test)
    aux = dict(loss_train=l_train, loss_test=l_test)
    return param_loss(params, x_test, y_test), aux


@jit
def outer_step(i, state, task):
    params = outer_get_params(state)
    g, aux = grad(maml_loss, has_aux=True)(params, task)
    return outer_opt_update(i, g, state), aux


# logging, plotting
log = Log(keys=['update', 'loss_train', 'loss_test'])
win_loss = None

_, params = net_init(rng=random.PRNGKey(42), input_shape=(-1, 1))
state = outer_opt_init(params)
for i, task_batch in tqdm(enumerate(taskbatch(task_fn=sinusoid_task,
                                              batch_size=args.task_batch_size,
                                              n_task=args.n_train_task,
                                              n_support=args.n_support))):
    state, aux = outer_step(i=i, state=state, task=(task_batch['x_train'],
                                                    task_batch['y_train'],
                                                    task_batch['x_test'],
                                                    task_batch['y_test']))
    log.append([(key, aux[key]) for key in ['loss_train', 'loss_test']] + [('update', i)])

    if (i + 1) % (args.n_train_task // args.task_batch_size // 100) == 0:
        if win_loss is None:
            win_loss = viz.line(
                X=log['update'],
                Y=onp.stack([log['loss_train'], log['loss_test']], axis=1),
                opts=dict(title=f'maml sinusoid regression',
                          xlabel='update',
                          ylabel='half l2 loss',
                          legend=['train', 'test'])
            )
        else:
            viz.line(
                X=log['update'],
                Y=onp.stack([log['loss_train'], log['loss_test']], axis=1),
                win=win_loss, update='replace'
            )

# visualize model
params = outer_get_params(state)
xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
targets = np.sin(xrange_inputs)
predictions = f(params, xrange_inputs)

win_inference = viz.line(Y=targets,
                         X=xrange_inputs,
                         name='target',
                         opts=dict(title='sinusoid inference',
                                   xlabel='x',
                                   ylabel='y',
                                   )
                         )
viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='append', name='pre-update predictions')

x1 = onp.random.uniform(low=-5., high=5., size=(args.n_support,1))
y1 = 1. * onp.sin(x1 + 0.)

for i in range(1, args.n_inner_step + 1):
    params, _ = inner_optimization(params, x1, y1, n_inner_step=1)
    predictions = f(params, xrange_inputs)
    viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='append', name=f'{i}-step predictions')
viz.line(Y=predictions, X=xrange_inputs, win=win_inference, update='replace', name=f'{i}-step predictions',
         opts=dict(legend=['target', 'pre-update predictions'] +
                          [f'{i}-step predictions' for i in range(1, args.n_inner_step + 1)]))

# serialize
np_dir = os.path.join(args.log_dir, 'np')
os.makedirs(np_dir, exist_ok=True)
onp.save(file=os.path.join(np_dir, f'log'), arr=log)
