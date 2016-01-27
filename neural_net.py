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

#################
# hyperparameters
#################
# learning rate
eta = 0.3
# momentum
alpha = 0.3
# number of hidden units
n = 4

###############
# function defs
###############
# sigmoid activation function for neurons
# The derivative of the sigmoid activation function is easily expressed in terms of the function itself:
# d sigma(z)/dz = sigma(z)x(1 - sigma(z))
# This is useful in deriving the back-propagation algorithm
# If derivative argument is true, return the derivative of the sigmoid
def sigmoid(z, derivative):
    if derivative:
        return sigmoid(z) * (1-sigmoid(z))
    else: # derivative is False
        return 1 / (1+np.exp(-z))

#################
# data structures
#################
# double check that order for attribute and target matrices are correct
# for ltr in letters_list_training[:6]: print ltr.value, ltr.attributes
# print "..."
# for ltr in letters_list_training[9995:]: print ltr.value, ltr.attributes
# print "---------"

# Training data as a 10000x17 matrix seeded with letter attributes
# Rows in data matrices correspond to number of items in minibatch
# columns correspond to values of these items (values of xi for all items X in training data)
# numpy stores data in row major order
X_attributes = np.full( (len(letters_list_training),16), [ltr.attributes for ltr in letters_list_training] )
# print X_attributes.shape
# print X_attributes[:6]
# print "..."
# print X_attributes[9995:]
# print "---------"

# save targets in the order entered into the matrix
X_targets = np.array([ list(ltr.value) for ltr in letters_list_training ])
# print X_targets[:6]
# print "..."
# ####print X_targets.shape
# print X_targets[9995:]
# print "---------"

# Preprocessing: scale the training data
# to have zero mean and unit variance along each column (feature)

#print X.mean(axis=0) # get mean of each column
# print X[1:].mean(axis=0)
# # standard deviation of columns
#print X_attributes[1:].std(0)
# # don't include column 1 (bias inputs) in the preprocessing
# X_scaled = (X - X[1:].mean(axis=0) / X[1:].std(0))
# print X_scaled
# # Note: getting divide by 0 errors with this method, using scikit implementation

# preprocessing using sklearn package, returns array
# scaled to be Gaussian with zero mean and unit variance
# only scale columns 2-17, don't include bias input column
X_scaled = preprocessing.scale(X_attributes)
#print X_attributes.std(axis=0)
#X_scaled = preprocessing.scale(X)
# X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
#print X_scaled

# Concatenate scaled data with the 1s needed for bias inputs
# put bias input at the end so we don't need to worry about indexing
# when going from hidden -> output layer
bias_input = np.full((len(letters_list_training), 1), 1.0)
###print bias_input.shape
X = np.concatenate((X_scaled, bias_input), axis=1)
#print X
###print X.shape

# transpose row vector to column vector
# by casting array to matrix then transposing
# print X[0,:].shape
# X_row = np.mat(X[0,:])
# print X_row.shape
# X_row.T
# X[0,:][np.newaxis, :].T
# X[0,:][None].T
#print X_row.shape

# The preprocessing module provides a utility class StandardScaler
# that implements the Transformer API to compute the mean and standard deviation
# on a training set so you can reapply the same transformation on the testing set
# see scikit-learn.org/stable/modules/preprocessing.html

# Initial weights matrix
# Weight matrices have the same number of columns as units in the previous layer
# and the same number of rows as units in the next layer
# n is the number of hidden units
initial_weights = np.random.uniform(low= -.25, high= .25, size=(n,17) )
# print initial_weights
###print initial_weights.shape

# output layer
# 4x26 matrix, for 4 units in hidden layer and 26 letters in the alphabet
# target for properly identified letter is .9, and the rest of the units should be .1
####Y = np.full( (4,26), np.random.uniform(low= .1, high= .9, size=(4,26) ) )
#print Y.shape #4x26

# weights from hidden layer to output layer
# 5 columns to allow for bias input (one column of 1s)
hidden_to_output_weights = np.random.uniform(low= -.25, high= .25, size=(26,5) )

# output matrix, 1 row by 26 columns for 26 letters of the alphabet
# only 1 row, only need one output (activations) for output layer
# don't initialize to anything
# target for properly identified letter is .9, and the rest of the units should be .1
Y = np.full((1, 26), None)

