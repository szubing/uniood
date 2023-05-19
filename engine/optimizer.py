from typing import Any, Dict, List, Set
import itertools

import torch


def build_optimizer(model,
                    optimizer_type,
                    base_lr,
                    weight_decay,
                    sgd_momentum=0.9,
                    backbone_multiplier=0.1,
                    clip_norm_value=0):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = base_lr
            weight_decay = weight_decay
            if "backbone" in key:
                lr = lr * backbone_multiplier
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            enable = clip_norm_value > 0.0

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_value)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim
        
        if optimizer_type == "sgd":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, base_lr, momentum=sgd_momentum
            )
        elif optimizer_type == "adamw":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, base_lr
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        
        return optimizer