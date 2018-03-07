# vae-pytorch
This is an implementation of Variational auto-encoder (VAE) based on the original paper [1].

[1] https://arxiv.org/pdf/1312.6114.pdf

## installation
This requires the following packages:

 - Python 3.6.5 or later
 - torch 0.3.1
 - torchvision 0.2.0
 - numpy 1.14.1
 - typing 3.6.2

In my environment, I installed pytorch via pip.

## train
To train models, including an encoder and a decoder, run

     $ python train.py

Also you can specify some arguments (i.e. ``-g'' for training with GPU if you have it).

## test
After the training, you can draw some samples with trained decoder by running

     $ python visualizer.py -m trained -n 100

where an argument ``-m'' specifies the location of trained model and ``-n'' also specifies the number of samples to draw.
