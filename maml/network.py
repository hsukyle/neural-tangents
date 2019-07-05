from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, Tanh, LogSoftmax)
from neural_tangents.layers import Dense, Conv


def select_activation(activation_name):
    if activation_name == 'relu':
        activation = Relu
    elif activation_name == 'tanh':
        activation = Tanh
    elif activation_name == 'identity':
        activation = Identity
    else:
        raise ValueError

    return activation


def convActivationNormLayer(n_filter, filter_shape, strides, padding, bias_coef, activation, norm):
    activation = select_activation(activation)
    layer = stax.serial(
        Conv(out_chan=n_filter, filter_shape=filter_shape, strides=strides, padding=padding, b_gain=bias_coef),
        activation
    )
    if norm == 'batch_norm':
        layer = stax.serial(
            layer,
            BatchNorm(axis=(0, 1, 2))  # normalize over N, H, W
        )
    elif norm is None or norm == 'None':
        pass
    else:
        raise ValueError

    return layer


def denseActivationNormLayer(n_hidden_unit, bias_coef, activation, norm):
    activation = select_activation(activation)
    layer = stax.serial(
        Dense(n_hidden_unit, b_gain=bias_coef),
        activation
    )
    if norm == 'batch_norm':
        layer = stax.serial(
            layer,
            BatchNorm(axis=0)
        )
    elif norm is None or norm == 'None':
        pass
    else:
        raise ValueError

    return layer


def mlp(n_output, n_hidden_layer, n_hidden_unit, bias_coef, activation='relu', norm=None):
    return stax.serial(
        *[denseActivationNormLayer(n_hidden_unit, bias_coef, activation, norm) for i in range(n_hidden_layer)],
        Dense(n_output, b_gain=bias_coef)
    )


def conv_net(n_output, n_conv_layer, n_filter, bias_coef, activation='relu', norm=None):
    return stax.serial(
        *[convActivationNormLayer(
            n_filter=n_filter,
            filter_shape=(3, 3),
            strides=(2, 2),
            padding='SAME',
            bias_coef=bias_coef,
            activation=activation,
            norm=norm
        ) for i in range(n_conv_layer)],
        Flatten,
        Dense(n_output, b_gain=bias_coef),
        # LogSoftmax
    )


if __name__ == '__main__':
    import jax.numpy as np
    from jax import random
    from jax.experimental import optimizers
    from jax import jit, grad
    from data import sinusoid_task, minibatch, omniglot_task, load_omniglot
    import matplotlib.pyplot as plt
    import os
    import ipdb
    from jax.experimental.stax import logsoftmax

    def sinusoid():

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


    def omniglot():
        n_way, n_support, n_query = 50, 15, 5
        net_init, f = conv_net(n_output=n_way,
                               n_conv_layer=4,
                               n_filter=64,
                               bias_coef=1,
                               activation='relu',
                               norm='None')
        _, params_init = net_init(rng=random.PRNGKey(42), input_shape=(-1, 28, 28, 1))

        def loss(params, batch):
            inputs, targets = batch
            logits = f(params, inputs)
            outputs = logsoftmax(logits)
            return -np.sum(outputs * targets) / targets.shape[0]

        def accuracy(params, batch):
            inputs, targets = batch
            target_class = np.argmax(targets, axis=-1)
            predicted_class = np.argmax(f(params, inputs), axis=-1)
            return np.mean(predicted_class == target_class)

        splits = load_omniglot(n_support=n_support, n_query=n_query)
        task = omniglot_task(splits['train'], n_way=n_way, n_support=n_support, n_query=n_query)

        opt_init, opt_update, get_params = optimizers.momentum(step_size=1e-0, mass=0.9)

        @jit
        def update(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, grad(loss)(params, batch), opt_state)

        opt_state = opt_init(params_init)

        n_update = 10000
        for i in range(n_update):
            opt_state = update(i, opt_state, (task['x_train'], task['y_train']))
            if i == 0 or (i + 1) % (n_update // 100) == 0:
                print(
                    i,
                    f"train loss: {loss(get_params(opt_state), (task['x_train'], task['y_train']))},"
                    f"\ttest loss: {loss(get_params(opt_state), (task['x_test'], task['y_test']))}"
                )
        trained_params = get_params(opt_state)


    omniglot()
