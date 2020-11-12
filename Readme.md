Pix2PixHD model
===============
The Pix2PixHD is one of the popular deep-learning methods for image translation of high-resolution images without significant artifacts. <br/>
Here, we modifed the code of the Pix2PixHD to use scientific datasets which have extensions of tif, npy, and fits. 

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
     
      # data option in the BaseOption class
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
      
>    And you have to set the pathes of input and target directories.

      # directory path for training in the TrainOption class
      --input_dir_train: './datasets/Train/Input' (default)
      --target_dir_train: './datasets/Train/Target' (default)
      
>    &

      # directory path for test in the TestOption class
      --input_dir_test: './datasets/Test/Input' (default)
      


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
* The size of height and width should be same. If the shape of your data is not square, you have to do cropping or padding the data before the training.
* The order of filenames is prepared to be in a sequence and should be same for the input and target data.

<br/>

**Training**    
 You can train the model by typing:

    python train.py

* When the model is training, it saves the model with specified intervals as a file with extension .pt or .pth at "./checkpoints/*dataset_name*/Model"
* You can set the saving frequency in *Pix2PixHD_Options.py*. If you define "save_freq" of 10000, for example, a file which have an extension .pt will be saved every 10000 iterations.
* It will save a pair of images for the Real data and Generated one by the model every specified interation at "./checkpoints/*dataset_name*/Image/Train"



<br/>
 
**Test**     
 You can generate data from the inputs by the model or test the model by typing:
 
    python test.py

* It will save the AI-generated data with specified intervals (: saving frequency) at "./checkpoints/*dataset_name*/Image/Test"
* When you set an iteration in TestOption class of *Pix2PixHD_Options.py*, it saves the generated data by a model which saved before.

<br/>
