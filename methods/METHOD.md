## Method building instructions

To add new method, we suggest by adding a single `your_method_name.py` into this directory. In `your_method_name.py`, one should implement a class inherited from the SourceOnly in `source_only.py`. There are some important notes for implementing your method class:

- You need to name your model backbone as `self.backbone` in the __init__ step of your method. Because we use such name to assign a different learning rate to backbone from other modules.

- Care to set `self.feature_dim` when different backbones may have different attribute name.

- You should explicitly to set `require_source` or `require_target` to be `True` or `False` in the __init__ step of your method. Doing so could also speed up the training process if your method do not need to use source or target data.

After have your `your_method_name.py`, add a new mehtod in the `__init__.py` for loading it.