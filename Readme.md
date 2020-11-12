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

Getting Started
------------

# Installation <br/>
* Install PyTorch and dependencies from http://pytorch.org

<br/>
# Dataset <br/>
* You need a large set of input & target image pairs for training.

<br/>
# Training <br/>

    python train.py


<br/>
# Testing <br/>
    
    python test.py
