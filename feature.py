import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets import dataset_classes
from datasets.transforms import build_transform
from datasets.utils import DatasetWrapper
from configs import default
from models import backbone_names, build_backbone, CLIP_MODELS
from models.partial_model import get_partial_model

import argparse

parser = argparse.ArgumentParser()

def main(args):
    # build backbone
    backbone = build_backbone(args.backbone).to(args.device)
    if args.backbone in CLIP_MODELS:
        backbone = backbone.visual
    backbone.eval()

    # partial model features
    if args.ft_last_layer:
        args.feature_dir = os.path.join(args.feature_dir, 'FT_LAST_LAYER')
        backbone, partial_model = get_partial_model(backbone, layer_idx=1, name=args.backbone)
    # extract features by different domains
    all_domains = dataset_classes[args.dataset].domains.keys()
    for domain in all_domains:
        save_feature_dir = os.path.join(args.feature_dir, f'features-imgAug_{args.image_augmentation}', args.backbone.replace('/',''), args.dataset)
        save_feature_path = os.path.join(save_feature_dir, f'{domain}.pth')
        if os.path.exists(save_feature_path):
            print(f'featuers already save in {save_feature_path}')
            continue
        else:
            os.makedirs(save_feature_dir, exist_ok=True)

        data = dataset_classes[args.dataset](args.data_dir, domain, domain)
        transform = build_transform(args.image_augmentation, args.backbone)

        data_loader = DataLoader(
                        DatasetWrapper(data.train, transform=transform),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=False,
                        pin_memory=torch.cuda.is_available(),
                    )

        with torch.no_grad():
            all_features = []
            all_paths = []
            for batch_datas in data_loader:
                batched_images = batch_datas['img'].to(args.device)
                batched_paths = batch_datas['impath']
                features = backbone(batched_images)
                all_features.append(features.data.cpu())
                all_paths += list(batched_paths)
            
        all_features = torch.cat(all_features)
        assert len(all_features) == len(all_paths)
        
        save_features = {}
        for i in range(len(all_features)):
            save_features[all_paths[i]] = all_features[i]
        
        torch.save(save_features, save_feature_path)
        print(f'features saved in -- {save_feature_path} -- done!')


if __name__ == "__main__":
    ###########################
    # Directory Config (modify if using your own paths)
    ###########################
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default.DATA_DIR,
        help="where the dataset is saved",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=default.FEATURE_DIR,
        help="where to save pre-extracted features",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-L/14@336px",
        choices=backbone_names,
        help="specify the encoder-backbone to use",
    )
    parser.add_argument(
        "--ft_last_layer",
        action="store_true",
        help="wheather only to finetune the last layer during training",
    )   
    parser.add_argument(
        "--dataset",
        type=str,
        default="office31",
        choices=dataset_classes.keys(),
        help="number of train shot",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for test (feature extraction and evaluation)",
    )
    parser.add_argument(
        "--image_augmentation",
        type=str,
        default='none',
        choices=['none', # only a single center crop
                'flip', # add random flip view
                'randomcrop', # add random crop view
                ],
        help="specify the image augmentation to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which cuda to be used",
    )
    args = parser.parse_args()
    main(args)