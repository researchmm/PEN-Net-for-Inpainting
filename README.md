# PEN-Net for Image Inpainting
![PEN-Net](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/master/docs/PEN-Net.gif?raw=true)

### [Arxiv Paper](https://arxiv.org/abs/1904.07475) | [Project](https://sites.google.com/view/1900zyh/pen-net) | [Poster](https://drive.google.com/open?id=1Zyfmqa6zUS4fd7aBg577WTPzJj0QyZM9) | [BibTex](https://github.com/researchmm/PEN-Net-for-Inpainting#citation)

Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting<br>
[Yanhong Zeng](https://sites.google.com/view/1900zyh),  [Jianlong Fu](https://jianlong-fu.github.io/), [Hongyang Chao](https://scholar.google.com/citations?user=qnbpG6gAAAAJ&hl),  and [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).<br>
In CVPR 2019.

<!-- ------------------------------------------------------------------------------ -->
## Introduction 
Existing inpainting works either fill missing regions by copying fine-grained image patches or generating semantically reasonable patches (by CNN) from region context, while neglect the fact that both visual and semantic plausibility are highly-demanded. 

Our proposals combine these two mechanisms by,
1) **Cross-Layer Attention Transfer (ATN).** We use the learned region affinity from high-lelvel feature maps to guide feature transfer in adjacent low-level layers in an encoder. 
2) **Pyramid Filling.** We fill holes multiple times (depends on the depth of the encoder) by using ATNs from deep to shallow. 

<!-- ------------------------------------------------------------------------------ -->
## Example Results 
We re-implement PEN-Net in Pytorch for faster speed, which is slightly different from the original Tensorflow version used in our paper. Each triad shows original image, masked input and our result.

![celebahq](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/pytorch/docs/celebahq.PNG?raw=true)
![dtd](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/pytorch/docs/dtd.PNG?raw=true)
![facade](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/pytorch/docs/facade.PNG?raw=true)
![places2](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/pytorch/docs/places2.PNG?raw=true)

<!-- -------------------------------------------------------- -->
## Run 

0. Requirements:
    * Install python3.6
    * Install [pytorch](https://pytorch.org/) (tested on Release 1.1.0)
1. Training:
    * Prepare training images filelist [[our split]](https://drive.google.com/open?id=1_j51UEiZluWz07qTGtJ7Pbfeyp1-aZBg)
    * Modify [celebahq.json](configs/celebahq.json) to set path to data, iterations, and other parameters.
    * Our codes are built upon distributed training with Pytorch.  
    * Run `python train.py -c [config_file] -n [model_name] -m [mask_type] -s [image_size] `. 
    * For example, `python train.py -c configs/celebahq.json -n pennet -m square -s 256 `
    * If you have any trouble with the configuration, please refer to configuration section.
2. Resume training:
    * Run `python train.py -n pennet -m square -s 256 `.
3. Testing:
    * Run `python test.py -c [config_file] -n [model_name] -m [mask_type] -s [image_size] `. 
    * For example, `python test.py -c configs/celebahq.json -n pennet -m square -s 256 `
4. Evaluating:
    * Run `python eval.py -r [result_path]`

<!-- ------------------------------------------------------------------- -->

## Configuration

### Training

Inside `configs/celebahq.json` or other custom JSON files,

- In `data_loader`
  - `name`: name of images.
  - `zip_root`: Root directory containing the zip folder with `name`
  - `flist_root`: Root directory containing several sub-directories with `name` whcih contain the corresponding `train.flist`, ... etc

For example, if we have

```
"data_loader": {
	"name": "celebahq",
	"zip_root": "datazip"
	"flist_root": "flist",
	...
}
```

Then we must have:

- `cd PEN-Net-for-Inpainting`
- `datazip/celebahq/celebahq.zip` where `celebahq.zip` is a zip folder containing input images.
- `flist/celebahq/train.flist` or other flist files which contain paths to images relative to the zip folder. In this case, e.g. `celebahq/celebahq1.png`

<!-- ------------------------------------------------------------------- -->

## Pretrained models
Download the models below and put it under `release_model/`

[CELEBA-HQ](https://drive.google.com/open?id=1Xf_LwP38PLL78817nXfpsHsLuCNWoPX9) | 
[DTD](https://drive.google.com/open?id=1OCrML2j6apv44-TxJvpOLOzavJmxZtJr) |
[Facade](https://drive.google.com/open?id=1cTcEMIuii3jJfc5sstXxMQNyTVcV3I8K) |
[Places2](https://drive.google.com/open?id=1Hd8DUCJMGnCZz53_19zV4_p06YJ0UDt2) 


We also provide more results of central square below for your comparisons 

[CELEBA-HQ](https://drive.google.com/open?id=13xa9Gf4q_tu7B3Wr97udBoTXWrMNkrtl) |
[DTD](https://drive.google.com/open?id=1GYsdh-0vZ-DS4MTR3nBfeAJjsqHS_qau) |
[Facade](https://drive.google.com/open?id=1g_2Wy2K4wpVaU1Sd3-XS9-jsmQgqJx5V) 

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
