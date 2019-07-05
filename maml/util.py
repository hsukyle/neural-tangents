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


class VisdomPlotter:
    def __init__(self, viz):
        self.viz = viz
        self.windows = {}

    def log_to_line(self, win_name, log, plot_keys, title, xlabel, ylabel, X, width=800, height=600, plot_smooth=False):
        if plot_smooth:
            Y = onp.stack([log[key] for key in plot_keys] +
                          [onp.convolve(log[key], [0.1] * 10, 'same') for key in plot_keys], axis=1)
            legend = plot_keys + [f'{key}_smooth' for key in plot_keys]
            dash = onp.array(['dot' for i in range(len(plot_keys) * 2)])
        else:
            Y = onp.stack([log[key] for key in plot_keys], axis=1)
            legend = plot_keys
            dash = onp.array(['dot' for i in range(len(plot_keys))])
        if win_name not in self.windows:
            self.windows[win_name] = self.viz.line(
                X=X, Y=Y,
                opts=dict(title=title, xlabel=xlabel, ylabel=ylabel, width=width, height=height,
                          legend=legend, dash=dash)
            )
        else:
            self.viz.line(
                X=X, Y=Y, win=self.windows[win_name], update='replace'
            )

    def line(self, win_name, Y, X, name, update=None, title=None, xlabel=None, ylabel=None, width=800, height=600):
        if win_name not in self.windows:
            self.windows[win_name] = self.viz.line(
                X=X, Y=Y, name=name, opts=dict(
                    title=title, xlabel=xlabel, ylabel=ylabel, width=width, height=height, showlegend=True
                )
            )
        else:
            self.viz.line(
                X=X, Y=Y, name=name, win=self.windows[win_name], update=update
            )
