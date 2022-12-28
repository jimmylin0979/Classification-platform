"""
CutMix-PyTorch
Reference : https://github.com/clovaai/CutMix-PyTorch

MIT License 

Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Optional
import torch
import numpy as np


class Mix(object):
    """ """

    def __init__(self, opts, use_cuda: Optional[bool] = True, *args, **kwargs) -> None:
        super(Mix, self).__init__()

        #
        self.use_cuda = use_cuda
        self.mixup_beta = getattr(opts, "mix.mixup_beta", 1.0)
        self.cutmix_beta = getattr(opts, "mix.cutmix_beta", 1.0)
        self.prob = getattr(opts, "mix.prob", 1.0)
        self.switch_prob = getattr(opts, "mix.switch_prob", 0.5)
        self.mode = getattr(opts, "mix.mode", "batch")

    def rand_bbox(self, size, lam):
        """ """

        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, inputs, targets, beta):
        """ """

        #
        batch_size = inputs.size()[0]
        if self.use_cuda:
            rand_index = torch.randperm(batch_size).cuda()
        else:
            rand_index = torch.randperm(batch_size)
        target_a = targets
        target_b = targets[rand_index]

        #
        lam = np.random.beta(beta, beta) if beta > 0 else 1.0
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2])
        )
        return inputs, target_a, target_b, lam

    def mixup(self, inputs, targets, beta):
        """
        @ beta: should larger than 0
        """

        #
        batch_size = inputs.size()[0]
        if self.use_cuda:
            rand_index = torch.randperm(batch_size).cuda()
        else:
            rand_index = torch.randperm(batch_size)

        #
        lam = np.random.beta(beta, beta) if beta > 0 else 1.0
        inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
        target_a, target_b = targets, targets[rand_index]
        return inputs, target_a, target_b, lam

    def forward(self, inputs, targets):
        """ """

        #
        mode = "none"
        prob = np.random.rand(1)
        if prob < self.prob:
            #
            swith_prob = np.random.rand(1)
            mode = "cutmix" if swith_prob < self.switch_prob else "mixup"

        #
        lam = None
        target_a, target_b = None, None
        if mode == "cutmix":
            inputs, target_a, target_b, lam = self.cutmix(
                inputs, targets, self.cutmix_beta
            )
        elif mode == "mixup":
            inputs, target_a, target_b, lam = self.mixup(
                inputs, targets, self.mixup_beta
            )
        else:
            # if mode == "none", do nothing to inputs
            # and set target_a as targets
            target_a = targets

        return mode, inputs, target_a, target_b, lam

    def mix_criterion(self, mode, criterion, preds, target_a, target_b, lam):
        """
        @ mode:
        """

        # Calculate loss via
        if mode == "cutmix" or mode == "mixup":
            loss = criterion(preds, target_a) * lam + criterion(preds, target_b) * (
                1.0 - lam
            )
        else:
            # no mix-based augmentation is joined, just simply do standard cross-entropy
            loss = criterion(preds, target_a)

        return loss
