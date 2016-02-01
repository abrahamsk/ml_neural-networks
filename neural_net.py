#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16

import sys
import math
import numpy as np
from input import letters_list_training, letters_list_testing
# preprocessing to scale training data
from sklearn import preprocessing
import random

# Neural network to recognize letters
# after training with the UCI machine learning repository.
# Network has 16 inputs (mapping to the 16 attributes of the letters in the data set)
# one layer of hidden units, and 26 output units (for 26 classes of letters)
#### Structures defined here used in experiment1.py

#################
# hyperparameters
#################

################
# Experiment 1 #
################
# learning rate
eta = 0.3
# momentum
alpha = 0.3
# number of hidden units
n = 4

# ################
# # Experiment 2 #
# ################
# # Test how changing the learning rate changes results
# # low learning rate
# eta = 0.5
# # high learning rate
# eta = 0.6
# # momentum
# alpha = 0.3
# # number of hidden units
# n = 4
#
# ################
# # Experiment 3 #
# ################
# # Test how changing the momentum changes results
# # learning rate
# eta = 0.3
# # low momentum
# alpha = 0.05
# # high momentum
# alpha = 0.6
# # number of hidden units
# n = 4
#
# ################
# # Experiment 4 #
# ################
# # Test how changing the number of hidden units changes results
# # learning rate
# eta = 0.3
# # momentum
# alpha = 0.3
# # smaller number of hidden units
# n = 4
# # larger number of hidden units
# n = 8

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

######################################################################################################

# double check that order for attribute and target matrices are correct
# for ltr in letters_list_training[:6]: print ltr.value, ltr.attributes
# print "..."
# for ltr in letters_list_training[9995:]: print ltr.value, ltr.attributes
# print "---------"

#### Training data as a 10000x17 matrix seeded with letter attributes ####
# Rows in data matrices correspond to number of items in minibatch
# columns correspond to values of these items (values of xi for all items X in training data)
# numpy stores data in row major order
X_attributes = np.full( (len(letters_list_training),16), [ltr.attributes for ltr in letters_list_training] )
# print X_attributes.shape
# print X_attributes[:6]
# print "..."
# print X_attributes[9995:]
# print "---------"

#### save targets in the order entered into the matrix ####
X_targets = np.array([list(ltr.value) for ltr in letters_list_training])
# print X_targets[:6]
# print "..."
# print X_targets.shape
# print X_targets[9995:]
# print "---------"

## Replaced this preprocessing with scikit implementation, getting div/0 errors
# print X.mean(axis=0) # get mean of each column
# print X[1:].mean(axis=0)
# # standard deviation of columns
#print X_attributes[1:].std(0)
# # don't include column 1 (bias inputs) in the preprocessing
# X_scaled = (X - X[1:].mean(axis=0) / X[1:].std(0))
# print X_scaled
##

#### preprocessing input using sklearn package, returns array ####
# scaled to be Gaussian with zero mean and unit variance along each column (feature)
X_scaled = preprocessing.scale(X_attributes)
# print "X attr std:", X_attributes.std(axis=0)
# print "X scaled std", X_scaled.std(axis=0)
# print "--------"
# print "X attr mean", X_attributes.mean(axis=0)
# print "X scaled mean", X_scaled.mean(axis=0)
# # X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
# print "X scaled:\n", X_scaled
#
# print "======================="


#### Concatenate scaled data with the 1s needed for bias inputs ####
# put bias input at the end so we don't need to worry about indexing [1:25]
# when going from hidden -> output layer
bias_input = np.full((len(letters_list_training), 1), 1.0)
# print bias_input.shape
X = np.concatenate((X_scaled, bias_input), axis=1)
# print "X:\n", X
# print X.shape #10000x17
# The preprocessing module provides a utility class StandardScaler
# that implements the Transformer API to compute the mean and standard deviation
# on a training set so you can reapply the same transformation on the testing set
# see scikit-learn.org/stable/modules/preprocessing.html

######################################################################################################

#### Testing data as a 10000x17 matrix seeded with letter attributes ####
# Rows in data matrices correspond to number of items in minibatch
# columns correspond to values of these items (values of xi for all items X in testing data)
X_test_attributes = np.full( (len(letters_list_testing),16), [ltr.attributes for ltr in letters_list_testing] )

#### save targets in the order entered into the matrix ####
X_test_targets = np.array([list(ltr.value) for ltr in letters_list_testing])

#### preprocessing input using sklearn package, returns array ####
# scaled to be Gaussian with zero mean and unit variance along each column (feature)
# Scale the test data using the μi and σi values
# computed from the training data (X_attributes), not the test data.

# scaler = preprocessing.StandardScaler().fit(X_attributes)
# X_test_scaled = scaler.transform(X_test_attributes)

scaler = preprocessing.StandardScaler().fit(X_attributes)
# print "scaler:", scaler
#StandardScaler(copy=True, with_mean=True, with_std=True)
# print "scaler mean:", scaler.mean_
# print "scaler scale", scaler.scale_
# print "scaler transform", scaler.transform(X_attributes)
X_test_scaled = scaler.transform(X_test_attributes)

# compare to X_scaled = preprocessing.scale(X_attributes)
# print "X test attr std:", X_test_attributes.std(axis=0)
# print "X test scaled std:", X_test_scaled.std(axis=0)
# print "--------"
# print "X test attr mean:", X_test_attributes.mean(axis=0)
# print "X test scaled mean", X_test_scaled.mean(axis=0)
# # print X_test_scaled.shape #10000x16
# print "X test scaled:\n", X_test_scaled #10000x16


#### Concatenate scaled data with the 1s needed for bias inputs ####
# put bias input at the end so we don't need to worry about indexing [1:25]
# when going from hidden -> output layer
test_bias_input = np.full((len(letters_list_testing), 1), 1.0)
# print test_bias_input.shape
X_test = np.concatenate((X_test_scaled, test_bias_input), axis=1)
# print X_test 10000x17

# print "X:\n", X
# print "X_test:\n", X_test

######################################################################################################

#### Weight matrix input -> hidden layer ####
# Weight matrices have the same number of columns as units in the previous layer
# and the same number of rows as units in the next layer
# n is the number of hidden units
input_to_hidden_weights = np.random.uniform(low= -.25, high= .25, size=(n, 17))
# print "input to hidden weights shape", input_to_hidden_weights.shape #4x17

#### Weights from hidden layer to output layer ####
# 5 columns to allow for bias input (one column of 1s)
hidden_to_output_weights = np.random.uniform(low= -.25, high= .25, size=(26,n+1) )

######################################################################################################

#### Output layer matrix, 1 row by 26 columns for 26 letters of the alphabet ####
# only 1 row, only need one output (activations) for output layer
# don't initialize to anything
# target for properly identified letter is .9, and the rest of the units should be .1
Y = np.full((1, 26), None)

