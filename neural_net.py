#!/usr/bin/env python

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16

import sys
import math
import numpy as np
from input import letters_list_training

# Neural network to recognize letters
# after training with the UCI machine learning repository.
# Network has 16 inputs (mapping to the 16 attributes of the letters in the data set)
# one layer of hidden units, and 26 output units (for 26 classes of letters)

# sigmoid activation function for neurons
def sigmoid(z):
  return 1 / (1+math.exp(-z)) # math.exp returns e**z

# training data as a 10000x17 matrix seeded with letter attributes
X = np.full( (len(letters_list_training),17), [ltr.bias_input_plus_attributes for ltr in letters_list_training] )
# print X.shape
# print X

# run training examples through neural net to train for letter recognition
#######
# main
#######
# increment = 0
# run = [1, 2, 3, 4, 5]
#
# for i in run:
#     text = "\rTesting instance "+str((increment)+1)+"/"+str(len(run))
#     sys.stdout.write(text)
#
    ## change weights after each training example

#     increment += 1