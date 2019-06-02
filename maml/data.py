import numpy as np
from jax import random

def sinusoid_task(n_support, n_query=None, amp_range=[0.1, 5.0], phase_range=[0.0, np.pi], input_range=[-5.0, 5.0]):
    if n_query is None:
        n_query = n_support

    amp = np.random.uniform(low=amp_range[0], high=amp_range[1])
    phase = np.random.uniform(low=phase_range[0], high=phase_range[1])

    inputs = np.random.uniform(low=input_range[0], high=input_range[1], size=(n_support+n_query, 1))
    targets = amp * np.sin(inputs - phase)

    return dict(x_train=inputs[:n_support], y_train=targets[:n_support], x_test=inputs[n_support:], y_test=targets[n_support:], amp=amp, phase=phase)

def minibatch(x_train, y_train, batch_size, train_epochs):
    """Generate minibatches of data for a set number of epochs."""
    epoch = 0
    start = 0
    key = random.PRNGKey(0)

    while epoch < train_epochs:
        end = start + batch_size

        if end > x_train.shape[0]:
            key, split = random.split(key)
            permutation = random.shuffle(split, np.arange(x_train.shape[0], dtype=np.int64))
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            epoch = epoch + 1
            start = 0
            continue

        yield x_train[start:end], y_train[start:end]
        start = start + batch_size


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import os

    output_dir = os.path.expanduser('~/code/neural-tangents/output')

    task = sinusoid_task(n_support=10)

    plt.scatter(x=task['x_train'], y=task['y_train'], c='b', label='train')
    plt.scatter(x=task['x_test'], y=task['y_test'], c='r', label='test')

    x_true = np.linspace(-5, 5, 1000)
    y_true = task['amp'] * np.sin(x_true - task['phase'])
    plt.plot(x_true, y_true, 'k-', linewidth=0.5, label='true')

    plt.legend()
    plt.savefig(fname=os.path.join(output_dir, 'sinusoid_task.png'))