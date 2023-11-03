import os

from tools.utils import check_isfile

def read_split_from_txt(txt_path, path_prefix):
    """Get a list of dict from txt file.

    Each dict includes:
    {'impath': str,
    'label': int,
    'classname': str}
    """
    with open(txt_path) as f:
        lst = []
        for ind, x in enumerate(f.readlines()):
            item = {}
            impath = x.split(' ')[0]
            impath = os.path.join(path_prefix, impath)
            check_isfile(impath)
            label = x.split(' ')[1].strip()
            classname = impath.split('/')[-2].replace('_', ' ')
            item = {'impath': impath,
                    'label': int(label),
                    'classname': classname}
            lst.append(item)

    return lst


def filter_split(data_source, label_filter):
    """Filter datas by label_filter

    Args:
        data_source (list): a list of Datum objects.
    """
    lst = []
    for item in data_source:
        if label_filter(item['label']):
            lst.append(item)
    return lst


def get_num_classes(data_source):
    """Count number of classes.

    Args:
        data_source (list): a list of Datum objects.
    """
    label_set = set()
    for item in data_source:
        label_set.add(item['label'])
    return max(label_set) + 1


def get_lab2cname(data_source, num_classes=None):
    """Get a label-to-classname mapping (dict).

    Args:
        data_source (list): a list of dict.
    """
    container = set()
    for item in data_source:
        container.add((item['label'], item['classname']))
    mapping = {label: classname for label, classname in container} 
    labels = list(mapping.keys())
    labels.sort()
    if num_classes is not None and len(labels) != num_classes:
        labels = range(num_classes)
    classnames = [mapping[label] if label in mapping.keys() else '' for label in labels]
    return mapping, classnames


class Benchmark(object):
    """A benchmark that contains 
    1) training data
    2) validation data
    3) test data

    Each dataset of train/val/test is a list, where each item in the list includes a dict:
    {'impath': str,
     'label': int,
     'classname': str}
    """

    dataset_name = "" # e.g. imagenet, etc.

    def __init__(self, train=None, val=None, test=None):
        self.train = train  # labeled training data source
        self.val = val  # validation data source
        self.test = test  # test data source
        self.num_classes = get_num_classes(train)
        self.lab2cname, self.classnames = get_lab2cname(train, self.num_classes)
        self.train_labels = []
        for item in train:
            self.train_labels.append(item['label'])


class MultiDomainsBenchmark(Benchmark):
    """
    Assume classes across domains are the same.
    [0 1 ..................................................................... N - 1]
    |----common classes --||----source private classes --||----target private classes --|

    """

    domains = {} # e.g. {'dslr': dslr.txt}

    def __init__(self, source_total, target_total, n_share=None, n_source_private=None):
        total_num_classes = get_num_classes(source_total)
        if n_share is not None:
            assert n_source_private is not None
            n_target_private = total_num_classes - n_share - n_source_private
            assert n_target_private >= 0
            common_classes = [i for i in range(n_share)]
            source_private_classes = [i + n_share for i in range(n_source_private)]
            target_private_classes = [i + n_share + n_source_private for i in range(n_target_private)]

            self.source_classes = common_classes + source_private_classes
            self.target_classes = common_classes + target_private_classes

            train = filter_split(source_total, label_filter=lambda label: label in self.source_classes)
            test = filter_split(target_total, label_filter=lambda label: label in self.target_classes)
        else:
            train = source_total
            test = target_total

        super().__init__(train=train, val=None, test=test)