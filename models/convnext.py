#
import timm
import torch
from torch import nn, Tensor

#
from typing import Optional

#
from utils import logger

# Reference :
# 1. https://www.dounaite.com/article/6285443bac359fc9133c6731.html
# 2.

class convnext(nn.Module):
    def __init__(self, opts, cfg: Optional[dict] = None) -> None:
        super(convnext, self).__init__()

        """
        Available pretrained model name on timm with patter "*convnext*"

        ['convnext_atto', 'convnext_atto_ols', 'convnext_base', 'convnext_base_384_in22ft1k', 
        'convnext_base_in22ft1k', 'convnext_base_in22k', 'convnext_femto', 'convnext_femto_ols', 
        'convnext_large', 'convnext_large_384_in22ft1k', 'convnext_large_in22ft1k', 'convnext_large_in22k', 
        'convnext_nano', 'convnext_nano_ols', 'convnext_pico', 'convnext_pico_ols', 'convnext_small', 
        'convnext_small_384_in22ft1k', 'convnext_small_in22ft1k', 'convnext_small_in22k', 'convnext_tiny', 
        'convnext_tiny_384_in22ft1k', 'convnext_tiny_hnf', 'convnext_tiny_in22ft1k', 'convnext_tiny_in22k', 
        'convnext_xlarge_384_in22ft1k', 'convnext_xlarge_in22ft1k', 'convnext_xlarge_in22k']
        """
        """
        Pretrained model detail
        1. convnext_base : Params 87.6 M, MACs 15.38 GMac
        2. 
        """

        #
        self.model_name = getattr(opts, "model.model_name", "convnext_base")
        self.num_classes = getattr(opts, "dataset.num_classes", 1000)

        #
        self.cfg = cfg
        self.backbone = timm.create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes
        )
        logger.info("{0}".format(self.backbone.default_cfg))

        #
        print(self.backbone)

    def forward(self, x: Tensor) -> Tensor:
        #
        # x = self.backbone.forward_features(x)
        x = self.backbone(x)
        return x
