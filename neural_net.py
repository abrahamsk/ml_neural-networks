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

################
# Experiment 1 #
################

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
# Training data as a 10000x17 matrix seeded with letter attributes
# Rows in data matrices correspond to number of items in minibatch
# columns correspond to values of these items (values of xi for all items X in training data)
# numpy stores data in row major order
X_attributes = np.full( (len(letters_list_training),16), [ltr.attributes for ltr in letters_list_training] )
# print X_attributes.shape
##print X_attributes

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
# 5 columns to allow for bias input
hidden_to_output_weights = np.random.uniform(low= -.25, high= .25, size=(26,5) )

# output matrix
# don't initialize to anything
# target for properly identified letter is .9, and the rest of the units should be .1
Y = np.full((1, 26), None)

######
# main
######
# run training examples through neural net to train for letter recognition
# Classification with a two-layer neural network (Forward propagation)
# For two-layer networks (one hidden layer):
# I. For each test example:
#     1. Present input to the input layer.
#     2. Forward propagate the activations times the weights to each node in the hidden layer.
#     3. Forward propagate the activations times weights from the hidden layer to the output layer.
#     4. Interpret the output layer as a classification.
epoch_increment = 0
epoch = 5

for iter in xrange(epoch):
    text = "\rEpoch "+str((epoch_increment)+1)+"/"+str(epoch)
    sys.stdout.write(text)

    # for loops:
    # training set: epoch
        # input -> hidden layer
        # hidden layer -> output layer

    # iterate through data matrix to operate on individual training instances
    for row in X[0:2]:
        print "\n----New row in input matrix----"#, row #instance i vector
        # transpose row vector for matrix multiplication
        ##print row.shape
        X_row = np.mat(row)
        #print X_row.shape
        X_col = X_row.transpose()
        # # X[0,:][np.newaxis, :].T
        # # X[0,:][None].T
        ###print X_col.shape
        ##print X_col

        # forward propagation
        # initial run of data, use sigmoid activation function
        # pass in dot products of inputs and weights
        hidden_layer = sigmoid(np.dot(initial_weights, X_col), False)
        print hidden_layer # 4x1

        # hidden_layer matrix is the activation at the hidden layer
        # use hidden layer activations as input for the output layer

    # use hidden layer activations to get activations for output layer
    # for output_row in hidden_layer:
    #     print output_row.shape
    #     print "++++New row in output matrix++++"
    #     Y_row = np.mat(output_row)
    #     print Y_row.shape #1x26

        ###output_layer = sigmoid(np.dot(hidden_to_output_weights, Y_row), False)
        #print output_layer.shape #4x26

    # use hidden layer activations to get activations for output layer
    # append one row of 1s to hidden layer to allow for bias input
    bias_input_hidden_layer = np.full((1, 1), 1.0)
    #print bias_input_hidden_layer
    hidden_layer_concat = np.concatenate((hidden_layer, bias_input_hidden_layer), axis=0)
    #print hidden_layer_concat

    # matrix multiply (hidden layer) dot (weights from hidden -> output)
    output_layer = np.dot(hidden_to_output_weights, hidden_layer_concat)
    ####print output_layer.shape #26x1

    # apply sigmoid function to output layer
    # to get activations at output layer
    Y = sigmoid(output_layer, False)
    ####print "output results", Y
    ####print "Y shape", Y.shape #26x1

        # calculate error

        # change weights after each training example

    epoch_increment += 1

# list for target: list of 26 with one valued at .9 and the rest valued at .1