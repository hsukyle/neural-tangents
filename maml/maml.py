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
import numpy as onp

net_init, net_fn = mlp(n_output=1, n_hidden_layer=2, n_hidden_unit=40, activation='relu', norm=None)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
opt_update = jit(opt_update)

def loss(net_params, x, y):
    predictions = net_fn(net_params, x)
    return np.mean((y - predictions) ** 2)

def inner_update(net_params, x_train, y_train, inner_step_size=0.1, n_inner_step=1):
    params = net_params
    sgd_fn = lambda g, params: params - inner_step_size * g
    for i in range(n_inner_step):
        g = grad(loss)(params, x_train, y_train)
        params = tree_multimap(sgd_fn, g, params)
    return params

def maml_loss(net_params, x_train, y_train, x_test, y_test):
    params = inner_update(net_params, x_train, y_train)
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

K = 20
rng = random.PRNGKey(42)
_, net_params = net_init(rng, (-1, 1))
opt_state = opt_init(net_params)

np_maml_loss = []
for i, task_batch in enumerate(taskbatch(sinusoid_task, batch_size=8, n_task=240000, n_support=K)):
    opt_state, l = step(i, opt_state, 
        (task_batch['x_train'], task_batch['y_train'], task_batch['x_test'], task_batch['y_test'])
    )

    np_maml_loss.append(l)

    if i % 1000 == 0:
        print(f"iteration {i}:\tmaml loss: {l}")

output_dir = os.path.expanduser('~/code/neural-tangents/output')
plt.plot(np_maml_loss, 'b-', label='maml loss')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1, x2, 0, 10))
plt.legend()
plt.savefig(fname=os.path.join(output_dir, 'sinusoid_maml_training_curve.png'))

plt.close()

net_params = get_params(opt_state)
xrange_inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
targets = np.sin(xrange_inputs)
predictions = net_fn(net_params, xrange_inputs)
plt.plot(xrange_inputs, predictions, label='pre-update predictions')
plt.plot(xrange_inputs, targets, label='target')

x1 = onp.random.uniform(low=-5., high=5., size=(K,1))
y1 = 1. * onp.sin(x1 + 0.)

for i in range(1,5):
    net_params = inner_update(net_params, x1, y1)
    predictions = net_fn(net_params, xrange_inputs)
    plt.plot(xrange_inputs, predictions, label='{}-step predictions'.format(i))
plt.legend()
plt.savefig(fname=os.path.join(output_dir, 'sinusoid_maml_inference.png'))
