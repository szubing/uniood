# How to install datasets

We suggest putting all datasets under the same foulder of `$DATASETS`, such path is set in `configs.default.py`. The file structure looks like

```
$DATASETS/
|-- Office31/
|-- OfficeHome/
|-- VisDa2017/
|-- domain_net/
```
If you have some datasets already installed somewhere else, you can create symbolic links in `$DATASETS/dataset_name` that point to the original data to avoid duplicate download.

### Office31

- Create a folder named `Office31` under `$DATASETS`.
- Prepare domains following

```
Office31/
|-- amazon/
|-- dslr/
|-- webcam/
amazon.txt
dslr.txt
webcam.txt
```

- Copy `txts/Office31/*.txt` to the current folder.

### OfficeHome

- Create a folder named `OfficeHome` under `$DATASETS`.
- Prepare domains following

```
OfficeHome/
|-- Art/
|-- Clipart/
|-- Product/
|-- RealWorld/
Art.txt
Clipart.txt
Product.txt
RealWorld.txt
```

- Copy `txts/OfficeHome/*.txt` to the current folder.

### VisDa2017

- Create a folder named `VisDa2017` under `$DATASETS`.
- Prepare domains following

```
VisDa2017/
|-- train/
|-- validation/
syn.txt
real.txt
```

- Copy `txts/VisDa2017/*.txt` to the current folder.

### domain_net

- Create a folder named `domain_net` under `$DATASETS`.
- Prepare domains following

```
domain_net/
|-- painting/
|-- real/
|-- sketch/
painting_train.txt
real_train.txt
sketch_train.txt
```

- Copy `txts/domain_net/*.txt` to the current folder.

### Custom dataset

Building your Custom dataset is straightforward by inheriting the class of `datasets.benchmark.MultiDomainsBenchmark`