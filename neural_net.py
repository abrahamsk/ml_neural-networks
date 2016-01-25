#!/usr/bin/env python

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16

import sys
import math
import numpy as np
from input import letters_list_training
# preprocessing to scale training data
from sklearn import preprocessing
import random

# Neural network to recognize letters
# after training with the UCI machine learning repository.
# Network has 16 inputs (mapping to the 16 attributes of the letters in the data set)
# one layer of hidden units, and 26 output units (for 26 classes of letters)


###############
# function defs
###############
# sigmoid activation function for neurons
def sigmoid(z):
  return 1 / (1+math.exp(-z)) # math.exp returns e**z

#################
# data structures
#################
# Training data as a 10000x17 matrix seeded with letter attributes
# Rows in data matrices correspond to number of items in minibatch
# columns correspond to values of these items (values of xi for all items X in training data)
X_attributes = np.full( (len(letters_list_training),16), [ltr.attributes for ltr in letters_list_training] )
# print X_attributes.shape
##print X_attributes

# Preprocessing: scale the training data
# to have zero mean and unit variance along each column (feature)

#print X.mean(axis=0) # get mean of each column
# print X[1:].mean(axis=0)
# # standard deviation of columns
# print X[1:].std(0)
# # don't include column 1 (bias inputs) in the preprocessing
# X_scaled = (X - X[1:].mean(axis=0) / X[1:].std(0))
# print X_scaled
# # Note: getting divide by 0 errors with this method, using scikit implementation

# preprocessing using sklearn package, returns array
# scaled to be Gaussian with zero mean and unit variance
# only scale columns 2-17, don't include bias input column
X_scaled = preprocessing.scale(X_attributes)
#X_scaled = preprocessing.scale(X)
# X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
##print X_scaled

# Concatenate scaled data with the 1s needed for bias inputs
bias_input = np.full((len(letters_list_training), 1), 1.0)
##print bias_input
X = np.concatenate((bias_input, X_scaled), axis=1)
print X
#print X.shape

# The preprocessing module provides a utility class StandardScaler
# that implements the Transformer API to compute the mean and standard deviation
# on a training set so you can reapply the same transformation on the testing set
# see scikit-learn.org/stable/modules/preprocessing.html

# Initial weights matrix
# Weight matrices have the same number of rows as units in the previous layer
# and the same number of columns as units in the next layer
# initial_weights= np.full( (17,4), random.uniform(-.25, .25) )
initial_weights = np.random.uniform(low= -.25, high= .25, size=(17,4) )
# print initial_weights
# print initial_weights.shape






#######
# main
#######
# run training examples through neural net to train for letter recognition

# increment = 0
# run = [1, 2, 3, 4, 5]
#
# for i in run:
#     text = "\rTesting instance "+str((increment)+1)+"/"+str(len(run))
#     sys.stdout.write(text)
#
    ## change weights after each training example

#     increment += 1