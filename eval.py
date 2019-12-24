import cv2
import os
import sys
import math
import time
import json
import glob
import argparse
import urllib.request
from PIL import Image, ImageFilter
from numpy import random
import numpy as np

from core import metric as module_metric
from core.utils import set_device
from core.inception import InceptionV3
from core.metric import calculate_activation_statistics, calculate_frechet_distance

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-r', '--resume', required=True, type=str)
args = parser.parse_args()

dims = 2048
batch_size = 4
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = set_device(InceptionV3([block_idx]))

def main():
  real_names = list(glob.glob('{}/*_orig.png'.format(args.resume)))
  fake_names = list(glob.glob('{}/*_comp.png'.format(args.resume)))
  real_names.sort()
  fake_names.sort()
  # metrics prepare for image assesments
  metrics = {met: getattr(module_metric, met) for met in ['mae', 'psnr', 'ssim']}
  # infer through videos
  real_images = []
  fake_images = []
  evaluation_scores = {key: 0 for key,val in metrics.items()}
  for rname, fname in zip(real_names, fake_names):
    rimg = Image.open(rname)
    fimg = Image.open(fname)
    real_images.append(np.array(rimg))
    fake_images.append(np.array(fimg))
  # calculating image quality assessments
  for key, val in metrics.items():
    evaluation_scores[key] = val(real_images, fake_images)
  print(' '.join(['{}: {:6f},'.format(key, val) for key,val in evaluation_scores.items()]))
  
  # calculate fid statistics for real images 
  real_images = np.array(real_images).astype(np.float32)/255.0
  real_images = real_images.transpose((0, 3, 1, 2))
  real_m, real_s = calculate_activation_statistics(real_images, model, batch_size, dims)
  
  # calculate fid statistics for fake images
  fake_images = np.array(fake_images).astype(np.float32)/255.0
  fake_images = fake_images.transpose((0, 3, 1, 2))
  fake_m, fake_s = calculate_activation_statistics(fake_images, model, batch_size, dims)

  fid_value = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)
  print('FID: {}'.format(round(fid_value, 5)))
  print('Finish evaluation from {}'.format(args.resume))
  



if __name__ == '__main__':
  main()

      
