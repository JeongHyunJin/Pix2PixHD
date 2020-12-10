Pix2PixHD model
===============
 Pix2PixHD is one of the popular deep-learning methods for image translation of high-resolution images without significant artifacts. <br/>
Here, we have modifed the code of the Pix2PixHD to use scientific datasets which have extensions of .tif or .npy or .fits. 

* References <br/>
    Baseline of the Pix2PixHD model: [Wang et al. 2018](https://arxiv.org/abs/1711.11585) ([GitHub](https://github.com/NVIDIA/pix2pixHD)) <br/>
    Application of the Pix2PixHD to Science: [Jeong et al. 2020](https://iopscience.iop.org/article/10.3847/2041-8213/abc255) ([GitHub](https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms)), [Shin et al. 2020](https://iopscience.iop.org/article/10.3847/2041-8213/ab9085) ([GitHub](https://github.com/NoelShin/Ca2Mag))  <br/>


* Flowchart and structures of the Pix2PixHD <br/>

<p align="center">
<img src="https://user-images.githubusercontent.com/68056295/101707480-a3295100-3ace-11eb-8818-f73a21190799.png" width="90%" height="90%"></center>
</p>

   The Pix2PixHD consists of two major networks: one is a generative network (generator) and the other is a discriminative network (discriminator).
   The generator tries to generate realistic output from input, and the discriminator tries to distinguish the more realistic pair between a real pair and a generated pair.
   The model use multi-scale discriminators that have an identical network structure but operate at different image scales.
   It downsample the data of real and generated pair by a factor of 2 to create a data pyramid of multi-scales.
   The real pair consists of a real input and a real target.
   The generated pair consists of a real input and an output from the generator.<br/>

   While the model is training, both networks compete with each other and get an update at every step with loss functions.
   Loss functions are objectives that score the quality of results by the model, and the networks automatically learn that they are appropriate for satisfying a goal, i.e., the generation of realistic data.
   They are iterated until the assigned iteration, which is a sufficient number assuring the convergence of the model.


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
>    Before running the model, you have to check or adjust the options for the input and target datasets.
     
     # data option in BaseOption class
     --dataset_name: 'Pix2PixHD' (default)

     --input_ch: 1 (default)
     --target_ch: 1 (default)

     --data_size: 1024 (default)

     --logscale_input: False (default)
     --logscale_target: False (default)

     --saturation_lower_limit_input: -1 (default)
     --saturation_upper_limit_input: 1 (default)
     --saturation_lower_limit_target: -1 (default)
     --saturation_upper_limit_target: 1 (default)
      
>    And you have to set the pathes of input and target directories.

      # directory path for training in TrainOption class
      --input_dir_train: './datasets/Train/Input' (default)
      --target_dir_train: './datasets/Train/Target' (default)
      
>    &

      # directory path for test in TestOption class
      --input_dir_test: './datasets/Test/Input' (default)
      


<br/>

Getting Started
------------

**Installation**    
* Install Anaconada from https://docs.anaconda.com/anaconda/install/ (We use numpy, scipy, astropy, and, random modules)
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
   You can train the model with default options:

    python3 Pix2PixHD_Train.py

* When the model is training, it saves the model every step (a step: saving frequency) as a file with an extension .pt or .pth at "./checkpoints/*dataset_name*/Model"
* You can set the saving frequency in *Pix2PixHD_Options.py*. If you define "save_freq" of 10000, for example, a file which have an extension .pt will be saved every 10000 iterations.
* It will save a pair of images for the Real data and Generated one by the model every specified step at "./checkpoints/*dataset_name*/Image/Train". You can define the steps from "display_freq" in *Pix2PixHD_Options.py*

<br/>

   You can train the model with manually modified options as below:
   
   Ex 1)
   
    python3 Pix2PixHD_Train.py \
    --dataset_name 'EUV2Mag' \
    --data_size 1024 \
    --input_ch 3 \
    --logscale_input True \
    --saturation_lower_limit_input 0 \
    --saturation_upper_limit_input 200 \
    --saturation_lower_limit_target -3000 \
    --saturation_upper_limit_target 3000 \
    --input_dir_train '../Datasets/Train_data/Train_input' \
    --target_dir_train '../Datasets/Train_data/Train_output' \
    --n_epochs 100
    
<br/>
   
   Ex 2)
   
    python3 Pix2PixHD_Train.py \
    --dataset_name 'Map2Sim' \
    --data_size 256 \
    --input_dir_train 'D:/Train_input' \
    --target_dir_train 'D:/Train_output' \
    --batch_size 64 \
    --save_freq 100 \
    --n_epochs 100
    
<br/>
 
**Test**     
   You can generate data from the inputs by the model or test the model with default options:
 
    python3 Pix2PixHD_Test.py

* It will save the AI-generated data every step (a step: saving frequency) at "./checkpoints/*dataset_name*/Image/Test"
* When you set an iteration in TestOption class of *Pix2PixHD_Options.py*, it saves the generated data by a model which saved before.
* BaseOptions in *Pix2PixHD_Options.py* when you train the model and when you test the model should be same.


<br/>

   You can generate data or test the model with manually modified options as below:
   
   Ex 1)
   
    python3 Pix2PixHD_Test.py \
    --dataset_name 'EUV2Mag' \
    --data_size 1024 \
    --input_ch 3 \
    --logscale_input True \
    --saturation_lower_limit_input 0 \
    --saturation_upper_limit_input 200 \
    --saturation_lower_limit_target -3000 \
    --saturation_upper_limit_target 3000 \
    --input_dir_test '../Datasets/Test_data/Test_input' \
    --iteration 100000
    
<br/>

   Ex 2)
   
    python3 Pix2PixHD_Test.py \
    --dataset_name 'Map2Sim' \
    --data_size 256 \
    --input_dir_test 'D:/Test_input' \
    --batch_size 64 \
    --save_freq 100 \
    --n_epochs 100
<br/>

**Outputs**   
   It will make directories and save outputs as below:
    
    # Pix2PixHD_Train.py:
       ./chechpoints/{dataset_name}/Image/Train/{iteration_real.png}
       ./chechpoints/{dataset_name}/Image/Train/{iteration_fake.png}
       ./chechpoints/{dataset_name}/Model/{iteration_G.pt}
       ./chechpoints/{dataset_name}/Model/{iteration_D.pt}

    # Pix2PixHD_Test.py:
       ./chechpoints/{dataset_name}/Image/Test/{iteration}/{filename_AI.extension}

<br/>

Network architectures and Hyperparameters
------------

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can run this code by changing the hyperparameters of the Pix2PixHD.

<br/>

**Generator** 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  To be written... <br/>
 
      # data augmentation in BaseOption class
     --n_downsample: 4 (default)
     --n_residual: 9 (default)

<br/>

**Discriminator** 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To be written... <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The number of discriminator: n_D <br/>

     # data augmentation in BaseOption class
     --n_D: 2 (default)

<br/>

**Hyperparameters** 

* The Loss configuration of the Objective functions   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Total loss = ( cGAN loss ) + ( lambda_FM ) * ( Feature Matching loss )   <br/>

      # hyperparameters in TrainOption class
      --lambda_FM: 10 (default)
      --GAN_type: 'LSGAN' (default)

<br/>

* Optimizer    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Optimizer : Adam solver <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; momentum beta 1 parameter : beta1 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; momentum beta 2 parameter : beta2 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Learning rate : lr <br/>


      # hyperparameters in TrainOption class
      --beta1: 0.5 (default)
      --beta2: 0.999 (default)
      --lr: 0.0002 (default)

<br/>

* Initializer

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Initialize Weights in Convolutional Layers : normal distribution, mean : 0.0, standard deviation : 0.02   

<br/>
