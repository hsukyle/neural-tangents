from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, Tanh, LogSoftmax)
from neural_tangents.layers import Dense, Conv


def denseActivationNormLayer(n_hidden_unit, bias_coef, activation, norm):
    if activation == 'relu':
        activation = Relu
    elif activation == 'tanh':
        activation = Tanh
    elif activation == 'identity':
        activation = Identity
    else:
        raise ValueError

    if norm is None:
        return stax.serial(
            Dense(n_hidden_unit, b_gain=bias_coef),
            activation
        )
    elif norm == 'batch_norm':
        return stax.serial(
            Dense(n_hidden_unit, b_gain=bias_coef),
            activation,
            BatchNorm(axis=0)
        )
    else:
        raise ValueError


def mlp(n_output, n_hidden_layer, n_hidden_unit, bias_coef, activation='relu', norm=None):
    return stax.serial(
        *[denseActivationNormLayer(n_hidden_unit, bias_coef, activation, norm) for i in range(n_hidden_layer)],
        Dense(n_output, b_gain=bias_coef)
    )


if __name__ == '__main__':
    import jax.numpy as np
    from jax import random
    from jax.experimental import optimizers
    from jax import jit, grad
    from data import sinusoid_task, minibatch
    import matplotlib.pyplot as plt
    import os

    net_init, net_fn = mlp(n_output=1, n_hidden_layer=2, bias_coef=1.0, n_hidden_unit=40, activation='relu',
                           norm='batch_norm')

    rng = random.PRNGKey(42)
    in_shape = (-1, 1)
    out_shape, net_params = net_init(rng, in_shape)


    def loss(params, batch):
        inputs, targets = batch
        predictions = net_fn(params, inputs)
        return np.mean((predictions - targets) ** 2)


    opt_init, opt_update, get_params = optimizers.momentum(step_size=1e-2, mass=0.9)
    opt_update = jit(opt_update)


    @jit
    def step(i, opt_state, batch):
        params = get_params(opt_state)
        g = grad(loss)(params, batch)
        return opt_update(i, g, opt_state)


    task = sinusoid_task(n_support=1000, n_query=100)

    opt_state = opt_init(net_params)
    for i, (x, y) in enumerate(minibatch(task['x_train'], task['y_train'], batch_size=256, train_epochs=1000)):
        opt_state = step(i, opt_state, batch=(x, y))
        if i == 0 or (i + 1) % 100 == 0:
            print(
                f"train loss: {loss(get_params(opt_state), (task['x_train'], task['y_train']))},"
                f"\ttest loss: {loss(get_params(opt_state), (task['x_test'], task['y_test']))}"
            )
    #
    # output_dir = os.path.expanduser('~/code/neural-tangents/output')
    #
    # x_true = np.linspace(-5, 5, 1000).reshape(-1, 1)
    # y_true = task['amp'] * np.sin(x_true - task['phase'])
    # plt.plot(x_true, y_true, 'k-', linewidth=0.5, label='true')
    #
    # y_pred = net_fn(get_params(opt_state), x_true)
    # plt.plot(x_true, y_pred, 'b-', linewidth=0.5, label='network')
    #
    # plt.legend()
    # plt.savefig(fname=os.path.join(output_dir, 'sinusoid_single_task_regression.png'))
