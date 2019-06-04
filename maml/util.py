import numpy as onp
from jax.experimental import optimizers
from functools import partial

class Log(dict):
    def __init__(self, keys):
        for key in keys:
            self[key] = onp.array([])

    def append(self, keys_and_values):
        for (key, value) in keys_and_values:
            self[key] = onp.append(self[key], onp.array(value))


def select_opt(name, step_size):
    if name == 'sgd':
        optimizer = partial(optimizers.sgd, step_size=step_size)
    elif name == 'momentum':
        optimizer = partial(optimizers.momentum, step_size=step_size, mass=0.9)
    elif name == 'adam':
        optimizer = partial(optimizers.adam, step_size=step_size)
    else:
        raise ValueError
    return optimizer
