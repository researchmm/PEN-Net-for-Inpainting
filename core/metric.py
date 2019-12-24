import numpy as np
import math
import os
from scipy import linalg
import urllib.request
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
# from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from core.inception import InceptionV3




def compare_mae(img_true, img_test):
  img_true = img_true.astype(np.float32)
  img_test = img_test.astype(np.float32)
  return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def ssim(frames1, frames2):
  error = 0
  for i in range(len(frames1)):
    error += compare_ssim(frames1[i], frames2[i], multichannel=True, win_size=51)
  return error/len(frames1)

def psnr(frames1, frames2):
  error = 0
  for i in range(len(frames1)):
    error += compare_psnr(frames1[i], frames2[i])
  return error/len(frames1)

def mae(frames1, frames2):
  error = 0
  for i in range(len(frames1)):
    error += compare_mae(frames1[i], frames2[i])
  return error/len(frames1)




def get_activations(images, model, batch_size=64, dims=2048, cuda=True, verbose=False):
  """Calculates the activations of the pool_3 layer for all images.
  Params:
  -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                   must lie between 0 and 1.
  -- model       : Instance of inception model
  -- batch_size  : the images numpy array is split into batches with
                   batch size batch_size. A reasonable batch size depends
                   on the hardware.
  -- dims        : Dimensionality of features returned by Inception
  -- cuda        : If set to True, use GPU
  -- verbose     : If set to True and parameter out_step is given, the number
                   of calculated batches is reported.
  Returns:
  -- A numpy array of dimension (num images, dims) that contains the
     activations of the given tensor when feeding inception with the
     query tensor.
  """
  model.eval()

  d0 = images.shape[0]
  if batch_size > d0:
    print(('Warning: batch size is bigger than the data size. '
      'Setting batch size to data size'))
    batch_size = d0

  n_batches = d0 // batch_size
  n_used_imgs = n_batches * batch_size

  pred_arr = np.empty((n_used_imgs, dims))
  for i in range(n_batches):
    if verbose:
      print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
    start = i * batch_size
    end = start + batch_size

    batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
    batch = Variable(batch)
    if torch.cuda.is_available:
      batch = batch.cuda()
    with torch.no_grad():
      pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.shape[2] != 1 or pred.shape[3] != 1:
      pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
  if verbose:
    print(' done')

  return pred_arr


def calculate_activation_statistics(images, model, batch_size=64,
  dims=2048, cuda=True, verbose=False):
  """Calculation of the statistics used by the FID.
  Params:
  -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                   must lie between 0 and 1.
  -- model       : Instance of inception model
  -- batch_size  : The images numpy array is split into batches with
                   batch size batch_size. A reasonable batch size
                   depends on the hardware.
  -- dims        : Dimensionality of features returned by Inception
  -- cuda        : If set to True, use GPU
  -- verbose     : If set to True and parameter out_step is given, the
                   number of calculated batches is reported.
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the inception model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the inception model.
  """
  act = get_activations(images, model, batch_size, dims, cuda, verbose)
  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real
  tr_covmean = np.trace(covmean)

  return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
