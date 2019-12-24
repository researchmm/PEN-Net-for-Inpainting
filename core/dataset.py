import random
from random import shuffle
import os 
import math 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from core.utils import ZipReader


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train', level=None):
    super(Dataset, self).__init__()
    self.split = split
    self.level = level
    self.w, self.h = data_args['w'], data_args['h']
    self.data = [os.path.join(data_args['zip_root'], data_args['name'], i) 
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'.flist'), dtype=np.str, encoding='utf-8')]
    self.mask_type = data_args.get('mask', 'pconv')
    if self.mask_type == 'pconv':
      self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(2000, 12000)]
      if self.level is not None:
        self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(self.level*2000, (self.level+1)*2000)]
      self.mask = self.mask*(max(1, math.ceil(len(self.data)/len(self.mask))))
    else:
      self.mask = [0]*len(self.data)
    self.data.sort()
    
    if split == 'train':
      self.data = self.data*data_args['extend']
      shuffle(self.data)
    if debug:
      self.data = self.data[:100]

  def __len__(self):
    return len(self.data)
  
  def set_subset(self, start, end):
    self.mask = self.mask[start:end]
    self.data = self.data[start:end] 

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    # load image
    img_path = os.path.dirname(self.data[index]) + '.zip'
    img_name = os.path.basename(self.data[index])
    img = ZipReader.imread(img_path, img_name).convert('RGB')
    # load mask 
    if self.mask_type == 'pconv':
      m_index = random.randint(0, len(self.mask)-1) if self.split == 'train' else index
      mask_path = os.path.dirname(self.mask[m_index]) + '.zip'
      mask_name = os.path.basename(self.mask[m_index])
      mask = ZipReader.imread(mask_path, mask_name).convert('L')
    else:
      m = np.zeros((self.h, self.w)).astype(np.uint8)
      if self.split == 'train':
        t, l = random.randint(0, self.h//2), random.randint(0, self.w//2)
        m[t:t+self.h//2, l:l+self.w//2] = 255
      else:
        m[self.h//4:self.h*3//4, self.w//4:self.w*3//4] = 255
      mask = Image.fromarray(m).convert('L')
    # augment 
    if self.split == 'train': 
      img = transforms.RandomHorizontalFlip()(img)
      img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
      mask = transforms.RandomHorizontalFlip()(mask)
      mask = mask.rotate(random.randint(0,45), expand=True)
      mask = mask.filter(ImageFilter.MaxFilter(3))
    img = img.resize((self.w, self.h))
    mask = mask.resize((self.w, self.h), Image.NEAREST)
    return F.to_tensor(img)*2-1., F.to_tensor(mask), img_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item
