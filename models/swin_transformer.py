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


class swin_transformer(nn.Module):
    def __init__(self, opts, cfg: Optional[dict] = None) -> None:
        super(swin_transformer, self).__init__()

        #
        """
        Available pretrained model name on timm with patter "*swin*"
        
        ['swin_base_patch4_window7_224', 'swin_base_patch4_window7_224_in22k', 'swin_base_patch4_window12_384', 
        'swin_base_patch4_window12_384_in22k', 'swin_large_patch4_window7_224', 'swin_large_patch4_window7_224_in22k', 
        'swin_large_patch4_window12_384', 'swin_large_patch4_window12_384_in22k', 'swin_s3_base_224', 
        'swin_s3_small_224', 'swin_s3_tiny_224', 'swin_small_patch4_window7_224', 'swin_tiny_patch4_window7_224', 
        'swinv2_base_window8_256', 'swinv2_base_window12_192_22k', 'swinv2_base_window12to16_192to256_22kft1k', 
        'swinv2_base_window12to24_192to384_22kft1k', 'swinv2_base_window16_256', 'swinv2_cr_small_224', 
        'swinv2_cr_small_ns_224', 'swinv2_cr_tiny_ns_224', 'swinv2_large_window12_192_22k', 
        'swinv2_large_window12to16_192to256_22kft1k', 'swinv2_large_window12to24_192to384_22kft1k', 
        'swinv2_small_window8_256', 'swinv2_small_window16_256', 'swinv2_tiny_window8_256', 'swinv2_tiny_window16_256']
        """
        """
        Pretrained model detail
        1. swin_base_patch4_window7_224_in22k : Params 86.78 M, MACs 15.14 GMac
        2. swin_base_patch4_window12_384_in22k
        3. swinv2_base_window12to24_192to384_22kft1k : Params 86.93 M, MACs 34.06 GMac
        4. swinv2_base_window12_192_22k
        """
        self.model_name = getattr(
            opts, "model.model_name", "swin_base_patch4_window7_224_in22k"
        )
        self.num_classes = getattr(opts, "dataset.num_classes", 1000)

        #
        self.cfg = cfg
        self.backbone = timm.create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes
        )
        logger.info("{0}".format(self.backbone.default_cfg))

    def forward(self, x: Tensor) -> Tensor:
        #
        # x = self.backbone.forward_features(x)
        x = self.backbone(x)
        return x
