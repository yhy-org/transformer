Project Overview
This project involves training and testing image classification models on the CIFAR-100 dataset using CNN and Transformer architectures. The goal is to compare the performance of these models with various data augmentation techniques including CutMix.
﻿
Setup Instructions
Environment Setup:
Ensure Python 3.8+ is installed.
Ensure Anaconda is installed.
Ensure Pytorch is installed.
Install required packages: 
import paddle
import torch
import numpy
import matplotlib
pip install torch torchvision tensorboard.
﻿
Data Preparation:
CIFAR-100 will be automatically downloaded using torchvision datasets.
﻿
Training the Model
Configure Parameters:
Set hyperparameters in the script including batch size, learning rate, and number of epochs in the configuration section of the training script.
﻿
Run Training:
Execute the training script: transformer.py/Simple_CNN.py/deeper_CNN.py/vgg16.py
Monitor training progress with TensorBoard: tensorboard --logdir=runs.
Check the output for accuracy metrics and loss.
﻿
Additional Information
Data Augmentation: The training script includes CutMix augmentation along with other techniques to enhance model generalization.
Model Architectures: Details on the CNN and Transformer architectures are embedded in the script. They can be adjusted for experimentation.
Logging and Visualization: Training and validation metrics are logged and can be visualized in real-time using TensorBoard.
For further customization and advanced configurations, refer to the inline comments in the training and testing scripts.