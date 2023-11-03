from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.utils import DatasetWrapper, FeatureWrapper
from datasets.transforms import build_transform
from datasets import dataset_classes

def build_data_loaders(dataset, 
                       data_dir, 
                       source_domain, 
                       target_domain, 
                       n_share, 
                       n_source_private,
                       image_augmentation,
                       backbone,
                       no_balanced,
                       batch_size,
                       num_workers,
                       source_feature_path=None,
                       target_feature_path=None,
                       test_feature_path=None,
                       val_feature_path=None):
    data = dataset_classes[dataset](data_dir, source_domain, target_domain, n_share, n_source_private)
    transform = build_transform(image_augmentation, backbone)
    sampler = None
    if not no_balanced:
        freq = Counter(data.train_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in data.train_labels]
        sampler = WeightedRandomSampler(source_weights, len(data.train_labels))

    source_loader = DataLoader(
                DatasetWrapper(data.train, transform=transform) if source_feature_path is None else FeatureWrapper(data.train, source_feature_path),
                batch_size=batch_size,
                sampler=sampler,
                shuffle=True if sampler is None else None,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
            )
    
    target_loader = DataLoader(
                DatasetWrapper(data.test, transform=transform) if target_feature_path is None else FeatureWrapper(data.test, target_feature_path),
                batch_size=batch_size,
                sampler=None,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
            )
    
    test_loader = DataLoader(
                DatasetWrapper(data.test, transform=build_transform("none", backbone)) if test_feature_path is None else FeatureWrapper(data.test, test_feature_path),
                batch_size=batch_size,
                sampler=None,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
            )
    
    val_loader = None
    if data.val is not None:
        val_loader = DataLoader(
                    DatasetWrapper(data.val, transform=build_transform("none", backbone)) if val_feature_path is None else FeatureWrapper(data.val, val_feature_path),
                    batch_size=batch_size,
                    sampler=None,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=torch.cuda.is_available(),
                )
    else:
        val_loader = DataLoader(
                    DatasetWrapper(data.train, transform=build_transform("none", backbone)) if val_feature_path is None else FeatureWrapper(data.train, val_feature_path),
                    batch_size=batch_size,
                    sampler=None,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=torch.cuda.is_available(),
                )
    
    return source_loader, target_loader, test_loader, val_loader