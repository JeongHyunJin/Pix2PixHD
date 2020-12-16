# -*- coding: utf-8 -*-
"""
Pipeline of the Pix2PixHD Model

Created on Wed 12 Nov 2020

@author: Hyun-Jin Jeong (https://jeonghyunjin.com, jeong_hj@khu.ac.kr)

Reference:
1) https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms
2) https://iopscience.iop.org/article/10.3847/2041-8213/abc255
3) https://iopscience.iop.org/article/10.3847/2041-8213/ab9085
"""

#------------------------------------------------------------------------------

import os
from astropy.io import fits
from os.path import split, splitext
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate
from random import randint
from PIL import Image

#------------------------------------------------------------------------------
# [1] Preparing the Input and Target data sets

class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
                
        if opt.is_train:
            self.input_format = opt.data_format_input
            self.target_format = opt.data_format_target
            self.input_dir = opt.input_dir_train
            self.target_dir = opt.target_dir_train

            self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.' + self.input_format)))
            self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.' + self.target_format)))
            print(len(self.label_path_list), len(self.target_path_list))
        else:
            self.input_format = opt.data_format_input
            self.input_dir = opt.input_dir_test

            self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.' + self.input_format)))
            

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []

# [ Train data ] ==============================================================
        if self.opt.is_train:
            self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)

            self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            
# [ Train Input ] =============================================================
            if self.input_format in ["tif", "tiff"]:
                IMG_A0 = np.array(Image.open(self.label_path_list[index]))
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            elif self.input_format in ["fits", "fts", "fit"]:
                IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
            
             
            #--------------------------------------
            if len(IMG_A0.shape) == 3:
                IMG_A0 = IMG_A0.transpose(2, 0 ,1)

            #--------------------------------------
            if self.opt.logscale_input == True:
                IMG_A0[np.isnan(IMG_A0)] = 0.1
                IMG_A0[IMG_A0 == 0] = 0.1
                IMG_A0 = np.log10(IMG_A0)
            else:
                IMG_A0[np.isnan(IMG_A0)] = 0
            
            #--------------------------------------
            UpIA = np.float(self.opt.saturation_upper_limit_input)
            LoIA = np.float(self.opt.saturation_lower_limit_input)
            
            if self.opt.saturation_clip_input == True:
                label_array = (np.clip(IMG_A0, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            else:
                label_array = (IMG_A0-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
                
            #--------------------------------------
            label_array = self.__rotate(label_array)
            label_array = self.__pad(label_array, self.opt.padding_size)
            label_array = self.__random_crop(label_array)
            
            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
                
# [ Train Target ] ============================================================
            if self.input_format in ["tif", "tiff"]:
                IMG_B0 = np.array(Image.open(self.target_path_list[index]))
            elif self.target_format in ["npy"]:
                IMG_B0 = np.load(self.target_path_list[index], allow_pickle=True)
            elif self.target_format in ["fits", "fts", "fit"]:
                IMG_B0 = np.array(fits.open(self.target_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_target option. It has to be tif or npy or fits.")
            
            #--------------------------------------
            if len(IMG_B0.shape) == 3:
                IMG_B0 = IMG_B0.transpose(2, 0 ,1)
            
            #--------------------------------------
            if self.opt.logscale_target == True:
                IMG_B0[np.isnan(IMG_B0)] = 0.1
                IMG_B0[IMG_B0 == 0] = 0.1
                IMG_B0 = np.log10(IMG_B0)
            else:
                IMG_B0[np.isnan(IMG_B0)] = 0
            
            #--------------------------------------
            IMG_B0[np.isnan(IMG_B0)] = 0
            UpIB = np.float(self.opt.saturation_upper_limit_target)
            LoIB = np.float(self.opt.saturation_lower_limit_target)
            
            if self.opt.saturation_clip_target == True:
                target_array = (np.clip(IMG_B0, LoIB, UpIB)-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            else:
                target_array = (IMG_B0-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            
            #--------------------------------------
            target_array = self.__rotate(target_array)
            target_array = self.__pad(target_array, self.opt.padding_size)
            target_array = self.__random_crop(target_array)
            
            target_tensor = torch.tensor(target_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.


# [ Test data ] ===============================================================
        else:
# [ Test Input ] ==============================================================
            if self.input_format in ["tif", "tiff"]:
                IMG_A0 = np.array(Image.open(self.label_path_list[index]))       
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            elif self.input_format in ["fits", "fts", "fit"]:                    
                IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
            
            #--------------------------------------
            if len(IMG_A0.shape) == 3:
                IMG_A0 = IMG_A0.transpose(2, 0 ,1)

            #--------------------------------------
            if self.opt.logscale_input == True:
                IMG_A0[np.isnan(IMG_A0)] = 0.1
                IMG_A0[IMG_A0 == 0] = 0.1
                IMG_A0 = np.log10(IMG_A0)
            else:
                IMG_A0[np.isnan(IMG_A0)] = 0
            
            #--------------------------------------
            UpIA = np.float(self.opt.saturation_upper_limit_input)
            LoIA = np.float(self.opt.saturation_lower_limit_input)
            
            label_array = (np.clip(IMG_A0, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)

            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
            #--------------------------------------
            
            return label_tensor, splitext(split(self.label_path_list[index])[-1])[0]
        
        return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
                   splitext(split(self.target_path_list[index])[-1])[0]

#------------------------------------------------------------------------------
# [2] Adjust or Measure the Input and Target data sets
                   
    def __random_crop(self, x):
        x = np.array(x)
        x = x[self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
        return x

    @staticmethod
    def __pad(x, padding_size):
        if type(padding_size) == int:
            if len(x.shape) == 3:
                padding_size= ((0, 0), (padding_size, padding_size), (padding_size, padding_size))
            else:
                padding_size = ((padding_size, padding_size), (padding_size, padding_size))
        return np.pad(x, pad_width=padding_size, mode="constant", constant_values=0)

    def __rotate(self, x):
        return rotate(x, self.angle, reshape=False)

    @staticmethod
    def __to_numpy(x):
        return np.array(x, dtype=np.float32)

    def __len__(self):
        return len(self.label_path_list)
    
#------------------------------------------------------------------------------
