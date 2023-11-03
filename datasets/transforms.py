from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, RandomResizedCrop,
    RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode

from models import CLIP_MODELS


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


SIZE = (224, 224)
# Mode of interpolation in resize functions
INTERPOLATION = INTERPOLATION_MODES["bicubic"]
# Mean and std (default: CoOp)
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
# Random crop
CROP_PADDING = 0
# Random resized crop
RRCROP_SCALE = (0.5, 1.0)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def build_transform(image_augmentation,
                    backbone_name,
                    size=SIZE,
                    interpolation=INTERPOLATION,
                    pixel_mean=IMAGENET_PIXEL_MEAN,
                    pixel_std=IMAGENET_PIXEL_STD,
                    crop_padding=CROP_PADDING,
                    rrcrop_scale=RRCROP_SCALE):
    """Build transformation function.

    Args:
        image_augmentation (str): name of image augmentation method. If none, just use center crop.
    """
    clip_mode = backbone_name in CLIP_MODELS
    if clip_mode:
        pixel_mean = CLIP_PIXEL_MEAN
        pixel_std = CLIP_PIXEL_STD
        if backbone_name == 'RN50x4':
            size = (288, 288)
        elif backbone_name == 'RN50x16':
            size = (384, 384)
        elif backbone_name == 'ViT-L/14@336px':
            size = (336, 336)

    normalize = Normalize(mean=pixel_mean, std=pixel_std)

    if image_augmentation == "none":
        # center crop
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            CenterCrop(size=size),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "flip":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            CenterCrop(size=size),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "randomcrop":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            RandomCrop(size=size, padding=crop_padding),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "randomresizedcrop":
        transform = Compose([
            RandomResizedCrop(size=size, scale=rrcrop_scale, interpolation=interpolation),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == 'twoCrops':
        transform_ = Compose([
            RandomResizedCrop(size=size, scale=rrcrop_scale, interpolation=interpolation),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
        transform = TransformTwice(transform_)
    else:
        raise ValueError("Invalid image augmentation method: {}".format(image_augmentation))
        
    return transform