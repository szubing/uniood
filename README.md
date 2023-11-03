## Introduction
This repostitory contains code for the paper [Universal Domain Adaptation from Foundation Models: A Baseline Study](https://arxiv.org/pdf/2305.11092.pdf). It is also a code framework for implementing Universal Domain Adaptation (UniDA) methods. One could easily add new methods, backbones, and datasets from this code framework, which are respectively built on different directories of `methods`, `models`, `datasets`. Follow the [METHOD.md](methods/METHOD.md), [MODEL.md](models/MODEL.md), and [DATASET.md](datasets/DATASET.md) instructions to build each custom module if you need.

## Available algorithms
The [currently available methods](methods) are:

* Source Only (SO)

* Universal domain adaptation through self supervision (DANCE, [Saito et al., 2020](https://github.com/VisionLearningGroup/DANCE))

* Ovanet: One-vs-all network for universal domain adaptation (OVANet, [Saito et al., 2021](https://github.com/VisionLearningGroup/OVANet))

* Unified optimal transport framework for universal domain adaptation (UniOT, [Chang et al., 2022](https://github.com/changwxx/UniOT-for-UniDA))

*  Learning transferable visual models from natural language supervision (CLIP zero-shot, [Radford et al., 2021](https://github.com/openai/CLIP))

* Robust fine-tuning of zero-shot models (WiSE-FT, [Wortsman et al., 2022](https://github.com/mlfoundations/wise-ft))

* Multimodality helps unimodality: Cross-modal few-shot learning with multimodal models (CLIP cross-model, [Lin et al., 2023](https://github.com/linzhiqiu/cross_modal_adaptation))

* Universal Domain Adaptation from Foundation Models: A Baseline Study (CLIP distillation, [Bin Deng and Kui Jia, 2023](https://github.com/szubing/uniood))

## Available datasets
The [currently available datasets](datasets) are:

* Office31 ([Saenko et al., 2010](https://link.springer.com/chapter/10.1007/978-3-642-15561-1_16))

* OfficeHome ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))

* VisDA ([Peng et al., 2017](https://arxiv.org/abs/1710.06924))

* DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))


## Reproducing paper results
To reproduce the results of the paper, 

(1) first, you have to prepare the datasets following the instructions on [DATASET.md](datasets/DATASET.md);

(2) then, set you default paths in `configs/default.py`, and extract features by runing `scripts/feature.sh` in the first step to speed up the process and then run other scripts;

(3) finally, report the results by runing the `print_*.py` --> figures and latex tables are saved in corresponding directories.

## Acknowledgements
We thank [CLIP cross-model](https://github.com/linzhiqiu/cross_modal_adaptation) for providing the CLIP text templates, and the [OSR](https://github.com/sgvaze/osr_closed_set_all_you_need) for providing the OSCR evaluation code.

## Citation
If this repository helps you, please kindly cite the following bibtext:

```
@misc{deng2023universal,
      title={Universal Domain Adaptation from Foundation Models: A Baseline Study}, 
      author={Bin Deng and Kui Jia},
      year={2023},
      eprint={2305.11092},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```