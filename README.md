# PEN-Net for Image Inpainting
![PEN-Net](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/master/docs/PEN-Net.gif?raw=true)

### [Arxiv Paper](https://arxiv.org/abs/1904.07475) | [CVPR Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/) | [Poster](https://drive.google.com/open?id=1Zyfmqa6zUS4fd7aBg577WTPzJj0QyZM9) | [BibTex](https://github.com/researchmm/PEN-Net-for-Inpainting#citation)

Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting<br>
[Yanhong Zeng](),  [Jianlong Fu](https://jianlong-fu.github.io/), [Hongyang Chao](),  and [Baining Guo]().<br>
In CVPR 2019 (Poster).

<!-- ------------------------------------------------------------------------------ -->
## Introduction 
Existing image inpainting works either fill missing regions by copying fine-grained image patches or generating semantically reasonable patches (by CNN) from region context, while neglect the fact that both visual and semantic plausibility are highly-demanded. 

Our proposals combine these two mechanisms by,
1) **Cross-Layer Attention Transfer (ATN).** We use the learned region affinity from high-lelvel feature maps to guide feature transfer in adjacent low-level layers in an encoder. 
2) **Pyramid Filling.** We fill holes multiple times (depends on the depth of the encoder) by using ATNs from deep to shallow. 

<!-- ------------------------------------------------------------------------------ -->
## Example Results 
We re-implement PEN-Net in Pytorch for faster speed, which is slightly different from the original Tensorflow version used in our paper. Each triad shows original image, masked input and our result.

![celebahq](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/master/docs/celebahq.PNG?raw=true)
![dtd](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/master/docs/dtd.PNG?raw=true)
![facade](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/master/docs/facade.PNG?raw=true)
![places2](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/master/docs/places2.PNG?raw=true)

<!-- -------------------------------------------------------- -->
## Run 

0. Requirements:
    * Install python3.6
    * Install [pytorch](https://pytorch.org/) (tested on Release 1.1.0)
1. Training:
    * Prepare training images filelist [[our split]](https://drive.google.com/open?id=1_j51UEiZluWz07qTGtJ7Pbfeyp1-aZBg)
    * Modify [celebahq.json](configs/celebahq.json) to set path to data, iterations, and other parameters.
    * Our codes are built upon distributed training with Pytorch.  
    * Run `python train.py -c [config_file] -n [model_name] -m [mask_type] -s [image_size] `. For example, `python train.py -c configs/celebahq.json -n pennet -m square -s 256 `
2. Resume training:
    * Run `python train.py -n pennet -m square -s 256 `.
3. Testing:
    * Run `python test.py -c [config_file] -n [model_name] -m [mask_type] -s [image_size] `. For example, `python test.py -c configs/celebahq.json -n pennet -m square -s 256 `
4. Evaluating:
    * Run `python eval.py -r [result_path]`

<!-- ------------------------------------------------------------------- -->
## Pretrained models
[CELEBA-HQ](https://drive.google.com/open?id=1Xf_LwP38PLL78817nXfpsHsLuCNWoPX9) | 
[DTD](https://drive.google.com/open?id=1OCrML2j6apv44-TxJvpOLOzavJmxZtJr) |
[Facade](https://drive.google.com/open?id=1cTcEMIuii3jJfc5sstXxMQNyTVcV3I8K) |
[Places2](https://drive.google.com/open?id=1Hd8DUCJMGnCZz53_19zV4_p06YJ0UDt2) 

Download the model dirs and put it under `release_model/`


<!-- ------------------------------------------------------------------- -->
## TensorBoard
Visualization on TensorBoard for training is supported. 

Run `tensorboard --logdir release_model --port 6006` to view training progress. 

<!-- ------------------------------------------------------------------- -->
## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{yan2019PENnet,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1486--1494},
  year = {2019}
}
```

### License
Licensed under an MIT license.
