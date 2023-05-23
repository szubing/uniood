from configs import default
from datasets import dataset_classes
from models import backbone_names, head_names
from methods import method_classes
import argparse

parser = argparse.ArgumentParser()

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
    "--result_dir",
    type=str,
    default=default.RESULT_DIR,
    help="where to save experiment results",
)
# parser.add_argument(
#     "--config_dir",
#     type=str,
#     default=default.CONFIG_DIR,
#     help="where to load additional paras of a specific method which names of 'method.yaml'. \
#         Note that the paras in method.yaml are not overwritten with the common args.",
# )

###########################
# Method Config
###########################
parser.add_argument(
    "--method",
    type=str,
    default="SO",
    choices=method_classes.keys(),
    help="which method to run",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed number",
)
parser.add_argument(
    "--debug",
    type=float,
    default=None,
    help="Hyperparameters analysis of CLIP distillation method for setting temp values (0.0 - 1.0)",
)

###########################
# model Config (methods)
###########################

parser.add_argument(
    "--backbone",
    type=str,
    default="ViT-L/14@336px",
    choices=backbone_names,
    help="specify the encoder-backbone to use",
)
parser.add_argument(
    "--classifier_head",
    type=str,
    default="prototype",
    choices=head_names,
    help="classifier head architecture",
)
parser.add_argument(
    "--fixed_backbone",
    action="store_true",
    help="wheather fixed backbone during training",
)
parser.add_argument(
    "--fixed_BN",
    action="store_true",
    help="wheather fixed batch normalization layers during training",
)
parser.add_argument(
    "--save_checkpoint",
    action="store_true",
    help="wheather fixed batch normalization layers during training",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="which cuda to be used",
)

###########################
# Training Config (trainer.py)
###########################
# Dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="visda",
    choices=dataset_classes.keys(),
    help="number of train shot",
)
parser.add_argument(
    "--source_domain",
    type=str,
    default="syn",
    help="source domain name",
)
parser.add_argument(
    "--target_domain",
    type=str,
    default="real",
    help="target domain name",
)
parser.add_argument(
    "--n_share",
    type=int,
    default=6,
    help="the number of common classes between source and target domains",
)
parser.add_argument(
    "--n_source_private",
    type=int,
    default=3,
    help="the number of private classes of source domain",
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
    "--no_balanced",
    action="store_true",
    help="trains with no balanced sampling",
)

# Optimizer
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["adamw", "sgd"],
    help="optimizer"
)
parser.add_argument(
    '--base_lr', 
    type=float, 
    default=1e-2,
    help='base learning rate'
)
parser.add_argument(
    '--backbone_multiplier', 
    type=float, 
    default=0.1,
    help='backbone learning rate scaling factor'
)
parser.add_argument(
    '--weight_decay', 
    type=float, 
    default=5e-4,
    help='weight decay'
)
parser.add_argument(
    '--momentum',
    type=float, 
    default=0.9,
    help='sgd momentum'
)
parser.add_argument(
    '--clip_norm_value', 
    type=float, 
    default=0,
    help='clip gradient value'
)
parser.add_argument(
    "--max_iter",
    type=int,
    default=10000,
    help="max iters to run",
)

# lr scheduler
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="cosine",
    choices=["cosine", "linear", "constant"],
    help="optimizer type"
)
parser.add_argument(
    "--warmup_iter",
    type=int,
    default=50,
    help="warmup iter",
)
parser.add_argument(
    "--warmup_type",
    type=str,
    default="linear",
    choices=["linear", "constant"],
    help="warmup type"
)
parser.add_argument(
    '--warmup_min_lr', 
    type=float, 
    default=1e-5,
    help='warmup min lr'
)