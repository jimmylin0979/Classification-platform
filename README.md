# torch-Classification

## What is torch-Classification ? 
---

`torch-Classification` is a flexible pytorch repository for classification task only (currently),  
you can just clone this repository as your classification project base.

Since the components are well seperated, you can add your own model without the worry to affect other classification modules. 

I have used this repository as my porject base and complete many classification task with good score, hope this repo will be helpful to you.


## Getting Start
---

### Install

```bash
git clone https://github.com/jimmylin0979/torch-Classification.git
cd torch-Classification.git
pip install -r requirements.txt
```

### Training

Take swin transformer as example, we can start training a model follow the configuration wrote in `configs/swin_transformer.yaml`, and save the checkpoints into folder `results/swin`

```bash
python main.py --mode=train --config=configs/swin_transformer.yaml --save-dir=results/swin
```

### Evaluating

```bash
python main.py --mode=eval --save-dir=results/swin
```

## Roadmap
---

- [ ] Explain AI, like CAM, heatmap, and so on ...
- [ ] Support more and more SOTA models
- [ ] More friendly startup tutorial


## Acknowledge
---

+ [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
+ [SAM Optimizer](https://github.com/davda54/sam)
+ [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)
+ [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)
+ [ema-pytorch](https://github.com/lucidrains/ema-pytorch)
+ [torch-toolbox](https://github.com/PistonY/torch-toolbox)

## Cite
---

Please notice me if i miss cite any author.  


```bibtex
@inproceedings{foret2021sharpnessaware,
  title={Sharpness-aware Minimization for Efficiently Improving Generalization},
  author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=6Tm1mposlrM}
}
```

```bibtex
@inproceesings{pmlr-v139-kwon21b,
  title={ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks},
  author={Kwon, Jungmin and Kim, Jeongseop and Park, Hyunseo and Choi, In Kwon},
  booktitle ={Proceedings of the 38th International Conference on Machine Learning},
  pages={5905--5914},
  year={2021},
  editor={Meila, Marina and Zhang, Tong},
  volume={139},
  series={Proceedings of Machine Learning Research},
  month={18--24 Jul},
  publisher ={PMLR},
  pdf={http://proceedings.mlr.press/v139/kwon21b/kwon21b.pdf},
  url={https://proceedings.mlr.press/v139/kwon21b.html},
  abstract={Recently, learning algorithms motivated from sharpness of loss surface as an effective measure of generalization gap have shown state-of-the-art performances. Nevertheless, sharpness defined in a rigid region with a fixed radius, has a drawback in sensitivity to parameter re-scaling which leaves the loss unaffected, leading to weakening of the connection between sharpness and generalization gap. In this paper, we introduce the concept of adaptive sharpness which is scale-invariant and propose the corresponding generalization bound. We suggest a novel learning method, adaptive sharpness-aware minimization (ASAM), utilizing the proposed generalization bound. Experimental results in various benchmark datasets show that ASAM contributes to significant improvement of model generalization performance.}
}
```
```bibtex
@inproceedings{yun2019cutmix,
    title={CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features},
    author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year={2019},
    pubstate={published},
    tppubtype={inproceedings}
}
```