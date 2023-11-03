import faiss # must be first imported in the main.py --> I don't know why; this pacakage is used in engine.evalator for computing h3-score
import torch

from configs import parser
from engine.trainer import UniDaTrainer
from tools.utils import set_random_seed

torch.set_num_threads(1)


def main(args):
    set_random_seed(args.seed)
    trainer = UniDaTrainer(args)
    if args.eval_only:
        trainer.load(args.checkpoint)
    else:
        print('Begin Training...')
        trainer.train()

    print('Begin Testing...')
    trainer.test()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)