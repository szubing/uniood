import warnings
import torch
import numpy as np
import random
import pickle
import os
import torch
import json
import PIL


def set_random_seed(seed):
    '''Set random seed for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = os.path.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def makedirs(path, verbose=False):
    '''Make directories if not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if verbose:
            print(path + " already exists.")


def save_obj_as_pickle(pickle_location, obj):
    '''Save an object as pickle.'''
    pickle.dump(
        obj, open(pickle_location, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Save object as a pickle at {pickle_location}")


def load_pickle(pickle_location, default_obj=None):
    '''Load a pickle file.'''
    if os.path.exists(pickle_location):
        return pickle.load(open(pickle_location, 'rb'))
    else:
        return default_obj


def save_as_json(obj, fpath):
    '''Save an object as json.'''
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def load_json(json_location, default_obj=None):
    '''Load a json file.'''
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj


def collect_env_info():
    """Return env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str


def get_save_dir(path_dir, dataset, method, *args):
    file_name = ''
    for v in args:
        file_name += f'{v}_'
    file_name += '.json'
    file_name = file_name.replace('/', '')
    makedirs(os.path.join(path_dir, dataset, method))
    return os.path.join(path_dir, dataset, method, file_name)


def get_save_logits_dir(path_dir, black_box_type, dataset, source_domain, target_domain, 
                        n_share, n_source_private, seed, prefix='target_logits'):
    filename = os.path.join(path_dir,
                            prefix, 
                            f'{black_box_type}'.replace('/', ''),
                            f'{dataset}',)
    
    makedirs(os.path.join(filename))
    return os.path.join(filename, f'{source_domain}_{target_domain}_{n_share}_{n_source_private}_{seed}.pth')


def get_save_scores_dir(path_dir, black_box_type, dataset, source_domain, target_domain, 
                        n_share, n_source_private, seed, prefix='iid_scores'):
    filename = os.path.join(path_dir,
                            prefix, 
                            f'{black_box_type}'.replace('/', ''),
                            f'{dataset}',)
    
    makedirs(os.path.join(filename))
    return os.path.join(filename, f'{source_domain}_{target_domain}_{n_share}_{n_source_private}_{seed}.pth')


def get_save_checkpoint_dir(path_dir, black_box_type, dataset, source_domain, target_domain, 
                        n_share, n_source_private, seed, prefix='checkpoints'):
    filename = os.path.join(path_dir,
                            prefix, 
                            f'{black_box_type}'.replace('/', ''),
                            f'{dataset}',)
    
    makedirs(os.path.join(filename))
    return os.path.join(filename, f'{source_domain}_{target_domain}_{n_share}_{n_source_private}_{seed}.pth')