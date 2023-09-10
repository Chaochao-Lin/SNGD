# Structural Natural Gradient Descent

This repository is the official implementation of [Reconstructing Deep Neural Networks: Unleashing the Optimization Potential of Natural Gradient Descent](). 


# Training

Examples of running the models:

### Dataset example 

MNIST： The training/validation/testing sets have 50,000/10,000/10,000 images, respectively.
Each sample is a gray scale image of size 28x28.

### Running Example 

#### Training the MLP model on the MNIST dataset:

- bash run_MLP_SNGD.sh

  or

- python test_mlp.py plain sngd a --lrate 1e-1

