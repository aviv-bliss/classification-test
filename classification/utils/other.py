import os
import shutil
from pathlib import Path
import random

def rename(path, copy_dir, pos=True):
    if not Path.isdir(copy_dir):
        os.mkdir(copy_dir)

    if pos:
        prefix = 'pos'
    else:
        prefix = 'neg'

    for i, filename in enumerate(sorted(os.listdir(path))):
        filename_new = '{}_{}.jpg'.format(prefix, i)
        shutil.copy(path/filename, copy_dir/filename_new)


def make_subsets(path, root_dir, n):
    train_dir = root_dir/'train'
    if not Path.isdir(train_dir):
        os.mkdir(train_dir)
    test_dir = root_dir/'test'
    if not Path.isdir(test_dir):
        os.mkdir(test_dir)
    val_dir = root_dir/'val'
    if not Path.isdir(val_dir):
        os.mkdir(val_dir)

    dir_lst = os.listdir(path)
    random.shuffle(dir_lst)

    for i, filename in enumerate(dir_lst):
        if i < n:
            shutil.copy(path / filename, test_dir / filename)
        if i >= n and i < 2*n:
            shutil.copy(path / filename, val_dir / filename)
        if i >= 2*n:
            shutil.copy(path / filename, train_dir / filename)





if __name__ == '__main__':
    # path = Path('/home/victoria/Documents/data/Agronabis/pos')
    # copy_dir = Path('/home/victoria/Documents/data/Agronabis/pos_copy')
    # rename(path, copy_dir, pos=True)

    path = Path('/home/victoria/Documents/data/Agronabis/pos_copy')
    root_dir = Path('/home/victoria/Documents/data/Agronabis')
    make_subsets(path, root_dir, n=20)