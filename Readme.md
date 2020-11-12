Pix2PixHD model
===============
The Pix2PixHD is one of the popular deep-learning methods for image translation of high-resolution images without significant artifacts.

* Reference <br/>
    Baseline of the Pix2PixHD model: [Wang_etal_2018](https://arxiv.org/abs/1711.11585) <br/>
    Application of the Pix2PixHD model to Science: [Jeong_etal_2020](https://iopscience.iop.org/article/10.3847/2041-8213/abc255), [Shin_etal_2020](https://iopscience.iop.org/article/10.3847/2041-8213/ab9085)  <br/>

<br/>

Environments
------------
This code has been tested on Ubuntu 18.04 with a Nvidia GeForce GTX Titan XP GPU, CUDA Version 11.0, Python 3.6.9, and PyTorch 1.3.1.

<br/>

Prerequisites
------------
* Linux or macOS
* Python 3
* NVIDIA GPU + CUDA cuDNN

<br/>

* Flags: see *Pix2PixHD_Options.py* for all the training and test flags.     
>    Before running the model, you have to check or adjust the options for the input and target dataset.
     
      # data option
      --dataset_name: 'Pix2PixHD' (default)
      
      --input_ch: 1 (default)
      --target_ch: 1 (default)
      
      --data_size: 1024 (default)
      
      --logscale_input: False (default)
      --logscale_target: False (default)
      
      --saturation_lower_limit_input: 1 (default)
      --saturation_upper_limit_input: 100 (default)
      --saturation_lower_limit_target: 1 (default)
      --saturation_upper_limit_target: 100 (default)
      

<br/>

Getting Started
------------

**Installation**    
* Install Anaconada from https://docs.anaconda.com/anaconda/install/
* Install PyTorch and dependencies from http://pytorch.org
* Install Pillow with pip or conda ( https://pillow.readthedocs.io/en/stable/installation.html )

<br/>

**Dataset**       
* You need a large set of input & target image pairs for training.
* The input and target should be same pixel size.
* The size of height and width should be same. If your datasets are not square, you have to do cropping the data or padding before the training.

<br/>

**Training**    

    python train.py


<br/>

**Test**     
    
    python test.py
