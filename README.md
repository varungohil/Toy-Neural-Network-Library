# nnlib

Toy neural network library implementation (nnlib) I used to explain automatic-differentiation implementation in a [machine learning lecture](https://youtu.be/CHCO1q2updI) at IIT Gandhinagar.

All the source of toy neural-network library is in nnlib directory.
Within nnlib directory, the layers directory has files defining classes for ReLU layer and Fully-Connected layer. The losses folder has a file defining MSE loss. The neural_net.py file has definition of NeuralNet class. This class calls upon the methods of other classes to  perform forward and backward propagation.

The test.py file defines and trains a toy fully connected neural network on sample dataset for a regression task.
