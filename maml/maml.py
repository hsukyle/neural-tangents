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
import wandb
import argparse
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sinusoid', \
    help='sinusoid or omniglot or miniimagenet')
parser.add_argument('--n_hidden_layer', type=int, default=2)
parser.add_argument('--n_hidden_unit', type=int, default=40)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--outer_step_size', type=float, default=1e-3)
parser.add_argument('--outer_opt_alg', type=str, default='adam', \
    help='adam or sgd or momentum')
parser.add_argument('--inner_opt_alg', type=str, default='sgd', \
    help='sgd or mse-exact or ode')
parser.add_argument('--inner_step_size', type=float, default=0.1)
parser.add_argument('--n_inner_step', type=int, default=1)
parser.add_argument('--task_batch_size', type=int, default=8)
parser.add_argument('--n_train_task', type=int, default=8*20000)
parser.add_argument('--n_support', type=int, default=20)
parser.add_argument('--log_dir', type=str, default=os.path.expanduser('~/code/neural-tangents/output'))
# parser.add_argument('--wandb_sync', type=bool, default=True)

args = parser.parse_args()

# if not args.wandb_sync:
#     os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(project='neural-tangents', dir=args.log_dir)
wandb.config.update(args)

net_init, net_fn = mlp(n_output=1, n_hidden_layer=args.n_hidden_layer, n_hidden_unit=args.n_hidden_unit, activation=args.activation, norm=args.norm)

if args.outer_opt_alg == 'adam':
    outer_opt = partial(optimizers.adam, step_size=args.outer_step_size)
else:
    raise ValueError

opt_init, opt_update, get_params = outer_opt()
opt_update = jit(opt_update)

def loss(net_params, x, y):
    predictions = net_fn(net_params, x)
    return np.mean((y - predictions) ** 2)

def inner_update(net_params, x_train, y_train, inner_step_size, n_inner_step):
    params = net_params
    sgd_fn = lambda g, params: params - inner_step_size * g
    for i in range(n_inner_step):
        g = grad(loss)(params, x_train, y_train)
        params = tree_multimap(sgd_fn, g, params)
    return params

def maml_loss(net_params, x_train, y_train, x_test, y_test):
    params = inner_update(net_params, x_train, y_train, args.inner_step_size, args.n_inner_step)
    return loss(params, x_test, y_test)

def batch_maml_loss(net_params, x_train_b, y_train_b, x_test_b, y_test_b):
    task_losses = vmap(partial(maml_loss, net_params))(x_train_b, y_train_b, x_test_b, y_test_b)
    return np.mean(task_losses)

@jit
def step(i, opt_state, task_batch):
    net_params = get_params(opt_state)
    g = grad(batch_maml_loss)(net_params, *task_batch)
    l = batch_maml_loss(net_params, *task_batch)
    return opt_update(i, g, opt_state), l

rng = random.PRNGKey(42)
_, net_params = net_init(rng, (-1, 1))
opt_state = opt_init(net_params)

for i, task_batch in enumerate(taskbatch(sinusoid_task, batch_size=args.task_batch_size, n_task=args.n_train_task, n_support=args.n_support)):
    opt_state, l = step(i, opt_state, 
        (task_batch['x_train'], task_batch['y_train'], task_batch['x_test'], task_batch['y_test'])
    )
    wandb.log({'iteration': onp.asarray(i), 'loss': onp.asarray(l)})
    if i % 1000 == 0:
        print(f"iteration {i}:\tmaml loss: {l}")

net_params = get_params(opt_state)
xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
targets = np.sin(xrange_inputs)
predictions = net_fn(net_params, xrange_inputs)
plt.plot(xrange_inputs, predictions, label='pre-update predictions')
plt.plot(xrange_inputs, targets, label='target')

x1 = onp.random.uniform(low=-5., high=5., size=(args.n_support,1))
y1 = 1. * onp.sin(x1 + 0.)

for i in range(1, 5):
    net_params = inner_update(net_params, x1, y1, inner_step_size=args.inner_step_size, n_inner_step=1)
    predictions = net_fn(net_params, xrange_inputs)
    plt.plot(xrange_inputs, predictions, label='{}-step predictions'.format(i))
plt.legend()
# plt.savefig(fname=os.path.join(output_dir, 'sinusoid_maml_inference.png'))
wandb.log({'sinusoid_inference_maml': wandb.Image(plt)})
plt.close()