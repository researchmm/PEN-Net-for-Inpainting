import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False, 
           func=tf.image.resize_bilinear, name='resize'):
  r""" resize feature map 
  https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/layers.py#L114
  """
  if scale == 1:
    return x
  if dynamic:
    xs = tf.cast(tf.shape(x), tf.float32)
    new_xs = [tf.cast(xs[1]*scale, tf.int32),
              tf.cast(xs[2]*scale, tf.int32)]
  else:
    xs = x.get_shape().as_list()
    new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
  with tf.variable_scope(name):
    if to_shape is None:
      x = func(x, new_xs, align_corners=align_corners)
    else:
      x = func(x, [to_shape[0], to_shape[1]], align_corners=align_corners)
  return x


@add_arg_scope
def AtnConv(x1, x2, mask=None, ksize=3, stride=1, rate=2, 
            softmax_scale=10., training=True, rescale=False):
  r""" Attention transfer networks implementation in tensorflow
  
  Attention transfer networks is introduced in publication:
    Learning Pyramid-Context Encoder Networks for High-Quality Image Inpainting, Zeng et al. 
    https://arxiv.org/pdf/1904.07475.pdf
    https://github.com/researchmm/PEN-Net-for-Inpainting

  inspired by:
    Generative Image Inpainting with Contextual Attention, Yu et al. 
    https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    https://arxiv.org/abs/1801.07892

  Args:
    x1:  low-level feature map with larger  size [b, h, w, c].
    x2: high-level feature map with smaller size [b, h/2, w/2, c].
    mask: Input mask, 1 for missing regions 0 for known regions. 
    ksize: Kernel size for attention transfer networks.
    stride: Stride for extracting patches from feature map.
    rate: Dilation for matching.
    softmax_scale: Scaled softmax for attention.
    training: Indicating if current graph is training or inference.
    rescale: Indicating if input feature maps need to be downsample 
  Returns:
    tf.Tensor: reconstructed feature map 
  """
  # downsample input feature maps if needed due to limited GPU memory
  if rescale:
    x1 = resize(x1, scale=1./2, func=tf.image.resize_nearest_neighbor)
    x2 = resize(x2, scale=1./2, func=tf.image.resize_nearest_neighbor)
  # get shapes
  raw_x1s = tf.shape(x1)
  int_x1s = x1.get_shape().as_list()
  int_x2s = x2.get_shape().as_list()
  # extract patches from low-level feature maps for reconstruction 
  kernel = 2*rate
  raw_w = tf.extract_image_patches(
    x1, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
  raw_w = tf.reshape(raw_w, [int_x1s[0], -1, kernel, kernel, int_x1s[3]])
  raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to [b, kernel, kernel, c, hw]
  raw_w_groups = tf.split(raw_w, int_x1s[0], axis=0)
  # extract patches from high-level feature maps for matching and attending
  x2_groups = tf.split(x2, int_x2s[0], axis=0)  
  w = tf.extract_image_patches(
    x2, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
  w = tf.reshape(w, [int_x2s[0], -1, ksize, ksize, int_x2s[3]])
  w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to [b, ksize, ksize, c, hw/4]
  w_groups = tf.split(w, int_x2s[0], axis=0)
  # resize and extract patches from masks
  mask = resize(mask, to_shape=int_x2s[1:3], func=tf.image.resize_nearest_neighbor)
  m = tf.extract_image_patches(
    mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
  m = tf.reshape(m, [1, -1, ksize, ksize, 1])
  m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to [1, ksize, ksize, 1, hw/4]
  m = m[0]
  mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)

  # matching and attending hole and non-hole patches 
  y = []
  scale = softmax_scale
  for xi, wi, raw_wi in zip(x2_groups, w_groups, raw_w_groups):
    # matching on high-level feature maps
    wi = wi[0]
    wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
    yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")
    yi = tf.reshape(yi, [1, int_x2s[1], int_x2s[2], (int_x2s[1]//stride)*(int_x2s[2]//stride)])
    # apply softmax to obtain attention score
    yi *=  mm  # mask
    yi = tf.nn.softmax(yi*scale, 3)
    yi *=  mm  # mask
    # transfer non-hole features into holes according to the atttention score
    wi_center = raw_wi[0]
    yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_x1s[1:]], axis=0), 
                                strides=[1, rate*stride, rate*stride, 1]) / 4.
    y.append(yi)
  y = tf.concat(y, axis=0)
  y.set_shape(int_x1s)
  # refine filled feature map after matching and attending 
  y1 = tf.layers.conv2d(y, int_x1s[-1]//4, 3, 1, dilation_rate=1, activation=tf.nn.relu, padding='SAME')
  y2 = tf.layers.conv2d(y, int_x1s[-1]//4, 3, 1, dilation_rate=2, activation=tf.nn.relu, padding='SAME')
  y3 = tf.layers.conv2d(y, int_x1s[-1]//4, 3, 1, dilation_rate=4, activation=tf.nn.relu, padding='SAME')
  y4 = tf.layers.conv2d(y, int_x1s[-1]//4, 3, 1, dilation_rate=8, activation=tf.nn.relu, padding='SAME')
  y = tf.concat([y1,y2,y3,y4],axis=3)
  if rescale:
    y = resize(y, scale=2., func=tf.image.resize_nearest_neighbor)
  return y
