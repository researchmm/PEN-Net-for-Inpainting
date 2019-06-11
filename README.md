# PEN-Net for Image Inpainting
![PEN-Net](https://github.com/researchmm/PEN-Net-for-Inpainting/docs/PEN-Net.gif)

### [Arxiv]() | 
Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting.<br>
[Yanhong Zeng](),  [Jianlong Fu](https://jianlong-fu.github.io/), [Hongyang Chao](),  and [Baining Guo]().<br>
In CVPR 2019 (Poster).

## Usage

```
# tensorflow version, when L=7
import tensorflow as tf
from layers.atnconv import AtnConv
cnum = 32
x = conv2d(input, cnum//2, ksize=3, stride=1)

# encode
enc_feats = []
dims = [cnum * i for i in [1, 2, 4, 8, 8, 8]]
for i in range(len(dims)):
  enc_feats.append(x)
  x = conv2d(x, dims[i], ksize=3, stride=2)
latent_feat = x

# attention transfer networks
attn_feats = []
x = latent_feat
for i in range(len(dims)):
  x = AtnConv(enc_feats[-(i+1)], x, mask)
  attn_feats.append(x)

# decode
x = latent_feat
dims = [cnum * i for i in [1./2, 1, 2, 4, 8, 8]]
for i in range(len(dims)):
  x = deconv2d(x, dims[-(i+1)], ksize=3, stride=2)
  x = tf.concat([x, attn_feats[i]], axis=3)

output = conv2d(x, 3, ksize=1, stride=1)

```


### Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{yan2019PENnet,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

### License
Licensed under an MIT license.