import numpy as np
from jax import random
from glob import glob
import imageio
import concurrent.futures
from tqdm import tqdm
import os
from collections import defaultdict


class Partition(object):
    def __init__(self, labels, n_shot, n_query):
        partition = defaultdict(list)
        cleaned_partition = {}
        for ind, label in enumerate(labels):
            partition[label].append(ind)
        for label in list(partition.keys()):
            if len(partition[label]) >= n_shot + n_query:
                cleaned_partition[label] = np.array(partition[label], dtype=np.int)
        self.partition = cleaned_partition
        self.subset_ids = np.array(list(cleaned_partition.keys()))

    def __getitem__(self, key):
        return self.partition[key]


def sinusoid_task(n_support, n_query=None, amp_range=[0.1, 5.0], phase_range=[0.0, np.pi], input_range=[-5.0, 5.0]):
    if n_query is None:
        n_query = n_support

    amp = np.random.uniform(low=amp_range[0], high=amp_range[1])
    phase = np.random.uniform(low=phase_range[0], high=phase_range[1])

    inputs = np.random.uniform(low=input_range[0], high=input_range[1], size=(n_support + n_query, 1))
    targets = amp * np.sin(inputs - phase)

    return dict(x_train=inputs[:n_support], y_train=targets[:n_support], x_test=inputs[n_support:],
                y_test=targets[n_support:], amp=amp, phase=phase)


def omniglot_task(split_dict, n_way, n_support, n_query=None):
    if n_query is None:
        n_query = n_support

    images, partition = split_dict['images'], split_dict['partition']

    sampled_subset_ids = np.random.choice(partition.subset_ids, size=n_way, replace=False)
    x_train, y_train, x_test, y_test = [], [], [], []
    for i, subset_id in enumerate(sampled_subset_ids):
        indices = np.random.choice(partition[subset_id], n_support + n_query, replace=False)
        x = images[indices]
        x = 1.0 - x.astype(np.float32) / 255.0  # invert black and white

        x_train.append(x[:n_support])
        x_test.append(x[n_support:])
        y_train.append(i * np.ones(n_support, dtype='int'))
        y_test.append(i * np.ones(n_query, dtype='int'))

    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train_int = np.concatenate(y_train, axis=0)
    y_test_int = np.concatenate(y_test, axis=0)

    assert y_train_int.ndim == y_test_int.ndim == 1
    y_train_one_hot = np.zeros([*y_train_int.shape, n_way])
    y_train_one_hot[np.arange(y_train_int.shape[0]), y_train_int] = 1
    y_test_one_hot = np.zeros([*y_test_int.shape, n_way])
    y_test_one_hot[np.arange(y_test_int.shape[0]), y_test_int] = 1

    return dict(x_train=x_train, y_train=y_train_one_hot, x_test=x_test, y_test=y_test_one_hot)


def taskbatch(task_fn, batch_size, n_task, **task_fn_kwargs):
    assert n_task % batch_size == 0
    i_task = 0
    while i_task < n_task:
        batch = [task_fn(**task_fn_kwargs) for i in range(batch_size)]
        result = dict()
        for key in batch[0].keys():
            val = []
            for task_dict in batch:
                val.append(task_dict[key])
            result[key] = np.stack(val, axis=0)
        yield result
        i_task += batch_size


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


def load_image(image_path):
    return imageio.imread(image_path), image_path


def load_omniglot(root_dir='/h/kylehsu/datasets/omniglot/omniglot_28x28/', n_support=5, n_query=15):
    image_paths = glob(os.path.join(root_dir, '**', '*.png'), recursive=True)

    class_name_to_paths = defaultdict(list)

    for image_path in image_paths:
        class_name = os.path.join(*image_path.split('/')[-3:-1])
        class_name_to_paths[class_name].append(image_path)

    classes = list(class_name_to_paths.keys())
    class_idx_to_name = {i: cls for i, cls in enumerate(classes)}
    class_name_to_idx = {cls: i for i, cls in enumerate(classes)}

    n_class_train, n_class_val, n_class_test = 1150, 50, 423
    assert len(classes) == n_class_train + n_class_val + n_class_test

    np.random.seed(42)
    idx_permutation = np.random.permutation(len(classes))

    split_to_class_idxs = dict(
        train=idx_permutation[:n_class_train],
        val=idx_permutation[n_class_train: n_class_train + n_class_val],
        test=idx_permutation[-n_class_test:]
    )
    splits = {}

    for split_name in split_to_class_idxs.keys():
        image_paths, class_idxs, class_names = [], [], []
        for class_idx in split_to_class_idxs[split_name]:
            class_image_paths = class_name_to_paths[class_idx_to_name[class_idx]]
            image_paths.extend(class_image_paths)
            class_idxs.extend([class_idx] * len(class_image_paths))
            class_names.extend([class_idx_to_name[class_idx]] * len(class_image_paths))
        image_paths, class_idxs, class_names = np.array(image_paths), np.array(class_idxs), np.array(class_names)
        splits[split_name] = dict(image_paths=image_paths, class_idxs=class_idxs, class_names=class_names)

    cache_path = os.path.join(root_dir, 'omniglot_cache.npy')

    if not os.path.isfile(cache_path):
        print(f'cache not found; loading images and saving cache to {cache_path}')
        for split_name, split_dict in tqdm(splits.items()):
            images = []
            for image_path in tqdm(split_dict['image_paths']):
                images.append(imageio.imread(image_path))
            images = np.stack(images, axis=0)
            images = np.expand_dims(images, axis=3)  # add a channel dimension
            split_dict['images'] = images
            splits[split_name] = split_dict

        np.save(file=cache_path, arr=splits)
    else:
        print(f'cache found; loading from {cache_path}')
        splits = np.load(file=cache_path, allow_pickle=True).item()

    for split_name, split_dict in splits.items():
        split_dict['partition'] = Partition(labels=split_dict['class_names'], n_shot=n_support, n_query=n_query)
        splits[split_name] = split_dict

    return splits


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from visdom import Visdom
    import ipdb

    output_dir = os.path.expanduser('~/code/neural-tangents/output')


    def visualize_one_task():

        task = sinusoid_task(n_support=10)

        plt.scatter(x=task['x_train'], y=task['y_train'], c='b', label='train')
        plt.scatter(x=task['x_test'], y=task['y_test'], c='r', label='test')

        x_true = np.linspace(-5, 5, 1000)
        y_true = task['amp'] * np.sin(x_true - task['phase'])
        plt.plot(x_true, y_true, 'k-', linewidth=0.5, label='true')

        plt.legend()
        plt.savefig(fname=os.path.join(output_dir, 'sinusoid_task.png'))


    def test_taskbatch():
        batch_size = 2
        for i, batch in enumerate(taskbatch(sinusoid_task, batch_size=batch_size, n_task=batch_size, n_support=50)):
            assert np.all(
                batch['amp'].reshape(-1, 1, 1) * np.sin(batch['x_train'] - batch['phase'].reshape(-1, 1, 1)) == batch[
                    'y_train'])
            assert np.all(
                batch['amp'].reshape(-1, 1, 1) * np.sin(batch['x_test'] - batch['phase'].reshape(-1, 1, 1)) == batch[
                    'y_test'])

            for i_task in range(batch_size):
                plt.scatter(batch['x_train'][i_task], batch['y_train'][i_task], label=f'task_{i_task + 1}_train')
                plt.scatter(batch['x_test'][i_task], batch['y_test'][i_task], label=f'task_{i_task + 1}_test')

        plt.legend()
        plt.savefig(fname=os.path.join(output_dir, 'sinusoid_task_batch.png'))


    def test_omniglot():
        viz = Visdom(port=8000, env='main')
        splits = load_omniglot()

        n_way, n_support, n_query = 3, 5, 7
        # task = omniglot_task(splits['train'], n_way=3, n_support=5, n_query=7)

        batch_size = 2
        for i, batch in enumerate(taskbatch(omniglot_task, batch_size=batch_size, n_task=batch_size,
                                            split_dict=splits['val'], n_way=n_way, n_support=n_support,
                                            n_query=n_query)):

            for i_task in range(batch_size):
                x_train = batch['x_train'][i_task]
                x_test = batch['x_test'][i_task]
                y_train = batch['y_train'][i_task]
                y_test = batch['y_test'][i_task]

                viz.images(tensor=np.transpose(x_train, (0, 3, 1, 2)),
                           nrow=n_support)
                viz.text(f'y_train: {y_train}')
                viz.images(tensor=np.transpose(x_test, (0, 3, 1, 2)),
                           nrow=n_query)
                viz.text(f'y_test: {y_test}')


    test_omniglot()
    # test_taskbatch()
