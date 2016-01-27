#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16

# import data structures, variables, and neural net from neural_net
# data structures in the global scope
from neural_net import *

###############
# function defs
###############
def forward_propagation():
    """
    Function called in train()
    Forward propagate the input through the neural network
    during neural network training
    Does not include error computation
    :return: output of neural net
    """
    # iterate through data matrix to operate on individual training instances
    # ---> using slices [0:2] to make running the program during debug quicker
    for row in X[0:3]:
        ####print "\n----New row in input matrix----"  # , row #instance i vector
        # transpose row vector for matrix multiplication
        # print row.shape #17,
        X_row = np.mat(row)
        # print X_row.shape
        X_col = X_row.transpose()
        # # X[0,:][np.newaxis, :].T
        # # X[0,:][None].T
        ###print X_col.shape
        ##print X_col

        # forward propagation
        # initial run of data, use sigmoid activation function
        # pass in dot products of inputs and weights
        hidden_layer = sigmoid(np.dot(initial_weights, X_col), False)
        print hidden_layer.shape  # 4x1
        print hidden_layer

        # hidden_layer is the activation at the hidden layer
        # use hidden layer activations as input for the output layer

    # use hidden layer activations to get activations for output layer
    # append one row of 1s to hidden layer to allow for bias input
    bias_input_hidden_layer = np.full((1, 1), 1.0)
    # print bias_input_hidden_layer
    hidden_layer_concat = np.concatenate((hidden_layer, bias_input_hidden_layer), axis=0)
    # print hidden_layer_concat

    # matrix multiply (hidden layer) dot (weights from hidden -> output)
    output_layer = np.dot(hidden_to_output_weights, hidden_layer_concat)
    ####print output_layer.shape #26x1

    # apply sigmoid function to output layer
    # to get activations at output layer
    Y = sigmoid(output_layer, False)
    ####print "output results", Y
    ####print "Y shape", Y.shape #26x1

    # return activation from output layers
    return Y


def back_propagation(output_activations, target):
    """
    Function called in train()
    The the back-propagation algorithm is used
    during training to update all weights in the network.
    Array of output layer activations and array of targets passed in
    Targets array located in neural_net file
    :return: error
    """
    # calculate error
    # calculate delta_k for each output unit
    # 2. Calculate the error terms:
    #   For each output unit k, calculate error term δk :
    #   δk ← ok(1 − ok)(tk − ok)
    #
    # o_k is output: 26 outputs
    # t_k is target: per training instance: if input is A,
    # then output for for matching node should be .9
    # the rest of the outputs should be .1
    #
    #   For each hidden unit j, calculate error term δj :
    #   δj ← hj(1−hj) ( (∑ k∈output units) wkj δk )

    # map target value to output node (e.g. A == node[0])

    # for k in output_activations:
    #     error = k*(1 - k)*(target - k)
    #
    #
    # return error


    # 3. change weights after each training example
    # For each weight wkj from the hidden to output layer:
    #   wkj ← wkj +Δwkj
    #   where
    #   Δwkj =ηδkhj
    #
    # For each weight wji from the input to hidden layer:
    #   wji ←wji +Δwji
    #   where
    #   Δwji =ηδjxi


# Training a multi-layer neural network
# Repeat for a given number of epochs or until accuracy on training data is acceptable:
# For each training example:
# 	1. Present input to the input layer.
# 	2. Forward propagate the activations times the weights to each node in the hidden layer.
# 	3. Forward propagate the activations times weights from the hidden layer to the output layer.
# 	4. At each output unit, determine the error E.
# 	5. Run the back-propagation algorithm to update all weights in the network.
def train(num_epochs):
    """
    train() calls forward_propagation() and back_propagation()
    Run training examples through neural net to train for letter recognition
    Classification with a two-layer neural network (Forward propagation)
    For two-layer networks (one hidden layer):
    # includes multiple epochs
     I. For each test example:
         1. Present input to the input layer.
         2. Forward propagate the activations times the weights to each node in the hidden layer.
         3. Forward propagate the activations times weights from the hidden layer to the output layer.
         4. Interpret the output layer as a classification.
    """
    epoch_increment = 0

    # run training for <num_epochs> number of epochs
    for iter in xrange(num_epochs):
        text = "\rEpoch "+str((epoch_increment)+1)+"/"+str(num_epochs)
        sys.stdout.write(text)

        # for loops:
        # training set: epoch
            # input -> hidden layer
            # hidden layer -> output layer

        # feedforward input through neural net: input layer -> hidden layer -> output
        # Y is the the output of the matrix, without any error correction
        # but already processed through the sigmoid function
        Y = forward_propagation()
        #print "Post feedforward call", Y.shape #26x1
        ####print "Post feedforward func, output Y", Y.shape

        # use back propagation to compute error and adjust weights
        # pass in activation of output layer
        back_propagation(Y, X_targets)

        epoch_increment += 1

    # used a list for targets: list of 26 with one valued at .9 and the rest valued at .1

################
# Experiment 1 #
################

######
# main
######
epochs = 5
#train the neural net for <epochs> number of epochs
# using forward and back propagation
train(epochs)
