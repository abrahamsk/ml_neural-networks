#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16


from __future__ import division # want float and not int division
# import data structures, variables, and neural net from neural_net
# data structures in the global scope
from neural_net import *
import string
import matplotlib.pyplot as plt


###############
# function defs
###############
def forward_propagation(row):
    """
    Function called in train()
    Forward propagate the input through the neural network
    during neural network training
    Does not include error computation
    :param row of data matrix:
    :return output of neural net:
    """

    ####print "\n----New row in input matrix----"  # , row #instance i vector
    # transpose row vector for matrix multiplication
    # print row.shape #17,
    X_row = np.mat(row)
    # print "X row shape", X_row.shape #1x17
    X_col = X_row.transpose()
    # # X[0,:][np.newaxis, :].T
    # # X[0,:][None].T
    # print "X col shape", X_col.shape #17x1
    ##print X_col

    # forward propagation
    # initial run of data, use sigmoid activation function
    # pass in dot products of inputs and weights
    hidden_layer = sigmoid(np.dot(input_to_hidden_weights, X_col), False)
    # print "hidden layer shape", hidden_layer.shape  # 4x1
    # print "hidden layer", hidden_layer
    # print hidden_layer.dtype # float64

    # hidden_layer is the activation at the hidden layer
    # use hidden layer activations as input for the output layer

    # use hidden layer activations to get activations for output layer
    # append one row of 1s to hidden layer to allow for bias input
    bias_input_hidden_layer = np.full((1, 1), 1.0)
    # print bias_input_hidden_layer
    hidden_layer_concat = np.concatenate((hidden_layer, bias_input_hidden_layer), axis=0)
    # print "hidden layer concat shape", hidden_layer_concat.shape #5x1
    # print "hidden layer in forward prop", hidden_layer_concat

    # matrix multiply (hidden layer) dot (weights from hidden -> output)
    output_layer = np.dot(hidden_to_output_weights, hidden_layer_concat)
    #print output_layer.shape #26x1

    # apply sigmoid function to output layer
    # to get activations at output layer
    Y = sigmoid(output_layer, False)
    # print "output results", Y
    # print "Y shape", Y.shape #26x1

    # return activations from hidden and output layers
    return hidden_layer, Y

################################################################################################

def back_propagation(hidden_activations, output_activations, target, row, alpha):
    """
    Function called in train()
    The the back-propagation algorithm is used
    during training to update all weights in the network.
    Pass in activation of output layer and
    target letter corresponding to the row that is currently being passed through the neural net
    :param hidden_activations, output_activations, target, row of input:
    :return error:
    """

    #### 2. Calculate the error terms ####
    #### calculate error delta_k for each output unit ####
    #   For each output unit k, calculate error term δk :
    #   δk ← ok(1 − ok)(tk − ok)
    #
    # o_k is output: 26 outputs
    # t_k is target: per training instance: if input is A,
    # then output for for matching node should be .9
    # the rest of the outputs should be .1

    # map target value to output node (e.g. A == node[0])
    target_unit = ltr_to_index[target.tostring()]
    # print "target unit:", target_unit

    # calculate target for each node
    # for node matching letter, t = .9, otherwise t = .1
    output_layer_targets = [.1 for i in range(0, 26)]
    output_layer_targets[target_unit] = .9
    # print output_layer_targets
    # print len(output_layer_targets)

    # list for errors at output layer and hidden layer
    output_layer_error = []
    hidden_layer_error = []
    # counters to move through nodes of output and hidden layers
    output_node_index = 0
    hidden_node_index = 0

    ### calculate error for each output layer node ###
    # use target list indices
    for k in range(len(output_activations)):
        # get the error at an individual node, using the place in the target list
        # that corresponds to the target for the individual node
        node_error = output_activations[k] * (1 - output_activations[k]) * (
            output_layer_targets[output_node_index] - output_activations[k])
        # print node_error
        # append this node's error to the list of output layer errors
        output_layer_error.append(node_error)
        output_node_index += 1 # move index of node forward by one
        # print output_error

    #### Calculate error for each hidden node ####
    # For each hidden unit j, calculate error term δj :
    # δj ← hj(1−hj) ( (∑ k∈output units) wkj δk )
    # h_j is activation of each hidden unit j
    for j in range(len(hidden_activations)):
        output_node_index = 0 # reset counter for use in summing
        output_sum = 0 # keeps track of (∑ k∈output units) wkj δk )
        # get the sum of weight[k][j]*node_error for all output units
        for k in range(len(output_activations)):
            #print hidden_to_output_weights[output_node_index-1][hidden_node_index]
            output_error = (output_activations[k] * (1 - output_activations[k]) * (
                output_layer_targets[output_node_index] - output_activations[k]))
            output_sum += (hidden_to_output_weights[output_node_index][hidden_node_index] * output_error)
            # print "------------------ output sum", output_sum
            output_node_index += 1
            # print output_node_index # prints 26 times before loop exits
        # replace for loop with matrix multiplication
        #np.dot(hidden_activations, hidden_to_output_weights)

        ## calculate error at an individual hidden node using the formula from class notes
        # including the output_sum (∑ k∈output units) wkj δk )
        # print "*************"
        # print "hidden_activations_concat[j]:", hidden_activations[j]
        # print "(1 - hidden_activations_concat[j]):", (1 - hidden_activations[j])
        # print "output sum:", output_sum
        hidden_node_error = hidden_activations[j] * (1 - hidden_activations[j]) * (output_sum)
        # print "hidden_node_error", hidden_node_error
        # print "*************"


        output_sum = 0 # reset for next hidden node
        # add this node's error to the list of errors for the hidden layer
        hidden_layer_error.append(hidden_node_error)
        # print hidden_node_error.shape #1x1
        # print "------------------------ hidden layer error + len", hidden_layer_error, len(hidden_layer_error)
        hidden_node_index += 1 # move index of hidden node forward by one
    # print output_layer_error
    # print len(output_layer_error) # len=26
    # print hidden_layer_error
    # print len(hidden_layer_error) # len=4

    #### 3. change weights after each training example ####
    # To avoid oscillations at large η, introduce momentum,
    # in which change in weight is dependent on past weight change:
    # Δw^t =η*δ_j*x_ji + αΔw^(t−1)_ji

    #### Change weights from hidden -> output layer ####
    # For each weight wkj from the hidden to output layer:
    #   wkj ← wkj +Δwkj
    #   where
    #   Δwkj =ηδkhj
    #counter to make sure all weights are being updated
    no_change = 0
    # print "\nhidden to output weights in backprop:\n", hidden_to_output_weights
    # print "hidden activations concat len: ", (len(hidden_activations_concat))
    # print "hidden activations concat:\n", hidden_activations_concat
    # print "output activations len", (len(output_activations))
    # print "output activations:\n", output_activations

    # save deltas for the next iteration of weight change
    # used in current iteration as the weight change from the previous iteration
    hidden_to_output_deltas = np.full((26, n+1), 0)
    # print hidden_to_output_deltas.shape #26x5
    for j in range(len(hidden_activations)):
        # print "j ", j
        for k in range(len(output_activations)):
            # print "\n"
            # print "k:", k #0 -> 25
            # print "j:", j #0 -> 3
            # print hidden_activations #4x1
            ###delta = eta * output_layer_error[k] * hidden_activations[j]
            delta = eta * output_layer_error[k]*hidden_activations[j] + alpha*hidden_to_output_deltas[k][j]
            # save deltas for the next iteration of weight change
            # print delta
            hidden_to_output_deltas[k][j] = delta

            # update weight
            # print "hidden to output weights shape", hidden_to_output_weights.shape #26x5
            hidden_to_output_weights_kj_prior = hidden_to_output_weights[k][j]
            hidden_to_output_weights[k][j] = hidden_to_output_weights[k][j] + delta
            # counter to make sure all weights are being updated
            if(hidden_to_output_weights[k][j] == hidden_to_output_weights_kj_prior):
                # print "no weight change"
                # print "k, j: ", k, j
                no_change += 1
    #check to make sure all weights are being updated
    # print "\nnum of weights unchanged hidden to output", no_change
    # print "hidden to output weights after change\n", hidden_to_output_weights
    # print "new weights hidden to output:\n", hidden_to_output_weights
######################
    # for j in range(len(hidden_activations)):
    # # print "j ", j
    #     for k in range(len(output_activations)):
    #         # print "k ", k
    #         delta = eta * output_layer_error[k] * hidden_activations[j]
    #         # update weight
    #         # print "hidden to output weights shape", hidden_to_output_weights.shape #26x5
    #         hidden_to_output_weights_kj_prior = hidden_to_output_weights[k][j]
    #         hidden_to_output_weights[k][j] = hidden_to_output_weights[k][j] + delta
    #         # counter to make sure all weights are being updated
    #         if (hidden_to_output_weights[k][j] == hidden_to_output_weights_kj_prior):
    #             # print "no weight change"
    #             # print "k, j: ", k, j
    #             no_change += 1
    # # check to make sure all weights are being updated
    # # print "\nnum of weights unchanged hidden to output", no_change
    # # print "hidden to output weights after change\n", hidden_to_output_weights
    # # print "new weights hidden to output:\n", hidden_to_output_weights
#####################

    #### Change weights from input -> hidden layer ####
    # # For each weight wji from the input to hidden layer:
    # #   wji ←wji +Δwji
    # #   where
    # #   Δwji =ηδjxi
    input_to_hidden_weights_ji_prior = 0
    # save deltas for the next iteration of weight change
    # used in current iteration as the weight change from the previous iteration
    input_to_hidden_deltas = np.full((len(hidden_activations), len(row)), 0)
    # print input_to_hidden_deltas.shape #4x17
    no_change_input_to_hidden_weight = 0
    # icount = 0
    # jcount = 0
    # print "len X", len(X) #len=10000
    # print "len hidden activations", len(hidden_activations) #len=4
    # for i in range(len(X[0:8])):
    for i in range(len(row)):
        # print "i ----------", icount
        # icount += 1
        # jcount = 0
        # print "hidden activations len in backprop:", (len(hidden_activations)) #len=5
        for j in range(len(hidden_activations)):
            # print X.shape #10000x17
            # print "j ", jcount
            # jcount += 1
            # print input_to_hidden_weights[j][i]
            # print "k ", k
            # weight delta = Δw^t =η*δ_j*x_ji + αΔw^(t−1)_ji
            # input_to_hidden_deltas[j][i] is the previous iteration's change in weights
            # print "\n-------\n"
            # print "delta", delta, "= eta", eta, "hidden_layer_error[j]", hidden_layer_error[j], "* X[j][i]", X[j][
            #     i], "+ alpha", alpha, "* input_to_hidden_deltas[j][i]", input_to_hidden_deltas[j][i]
            delta = eta * hidden_layer_error[j]*X[j][i] + alpha*input_to_hidden_deltas[j][i]
            # print delta
            # save deltas for the next iteration of weight change
            input_to_hidden_deltas[j][i] = delta
            # print "input to hidden delta:", input_to_hidden_deltas[j][i]
            # print input_to_hidden_weights.shape #4x17
            input_to_hidden_weights_ji_prior = input_to_hidden_weights[j][i]
            # print "prior weight", input_to_hidden_weights_ji_prior
            # print "delta:", delta
            # print "input_to_hidden_weights[j][i]:", input_to_hidden_weights[j][i]
            # # update weight:
            # print "input to hidden weights shape", input_to_hidden_weights.shape #4x17
            # print "input_to_hidden_weights[j][i]:", input_to_hidden_weights[j][i], " + delta: ", delta
            input_to_hidden_weights[j][i] = input_to_hidden_weights[j][i] + delta
            # print "new weight ", input_to_hidden_weights[j][i]

            # counter to make sure all weights are being updated
            if(input_to_hidden_weights[j][i] == input_to_hidden_weights_ji_prior):
                # print "no weight change"
                # print "j, i: ", j, i
                no_change_input_to_hidden_weight += 1
    #check to make sure all weights are being updated
    if(no_change_input_to_hidden_weight > 0):
        print "\nnum of weights unchanged input to hidden", no_change_input_to_hidden_weight
    # print "input to hidden weights after change\n", input_to_hidden_weights
    # print input_to_hidden_weights




################################################################################################

# Training a multi-layer neural network
# Repeat for a given number of epochs or until accuracy on training data is acceptable:
# For each training example:
# 	1. Present input to the input layer.
# 	2. Forward propagate the activations times the weights to each node in the hidden layer.
# 	3. Forward propagate the activations times weights from the hidden layer to the output layer.
# 	4. At each output unit, determine the error E.
# 	5. Run the back-propagation algorithm to update all weights in the network.
def train(num_epochs, alpha):
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

    training_acc_list = []
    testing_acc_list = []

    # run training for <num_epochs> number of epochs (defined before func is called in main)
    # each epoch runs through entire training set
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

        # iterate through data matrix to operate on individual training instances
        # ---> using slices to make running the program during debug quicker
        target_row = 0 # count keeps track of which index of target to pass in
        # for row in X[0:8]:
        for row in X[0:20]:
            hidden_layer = [] # list to hold hidden layer, to pass to back_propagation once it's filled
            hidden_layer, Y = forward_propagation(row)
            # print "Post feedforward call", Y.shape #26x1
            # print "Post feedforward func, output Y", Y
            # print "Hidden layer shape after forward prop", hidden_layer.shape

            # use back propagation to compute error and adjust weights
            # pass in activations of hidden and output layer and target letter corresponding to the row
            # that is currently being passed through the neural net
            # print X_targets[target_row]
            back_propagation(hidden_layer, Y, X_targets[target_row], row, alpha)

            # move to next row of input data to use new target
            target_row += 1

        # increment epoch after all input data is processed
        epoch_increment += 1
        # print "\nepoch incremented", epoch_increment

        # After each epoch, calculate the network's accuracy
        # on the training set and the test set
        # training_accuracy, testing_accuracy = calculate_accuracy(X[0:8], X_test[0:8], epoch_increment)
        training_accuracy, testing_accuracy = calculate_accuracy(X[0:20], X_test[0:20], epoch_increment)
        training_acc_list.append(training_accuracy)
        testing_acc_list.append(testing_accuracy)

        print "\ntraining list in train", training_acc_list
        print "testing list in train", testing_acc_list


    # print "done with epochs"
    return training_acc_list, testing_acc_list

    # use a list for targets: list of 26 with one valued at .9 and the rest valued at .1
    # use error to calculate updated weights

################################################################################################

def calculate_accuracy(training_data, test_data, epoch_num):
    """
    After each epoch, calculate the network's accuracy
    on the training set and the test set
    :param training_data, test_data, epoch_num
    :return training_accuracy, testing_accuracy:
    """
    # counters for neural net votes
    correct_train_vote = 0
    correct_test_vote = 0

    #### Training data ####
    # use forward_propagation(row) to calculate predictions for training data
    training_predictions = []
    # record values for plotting
    training_letter_vote = []
    training_letter_actual = []
    target_row = 0
    for row in training_data:
        hidden_layer, Y_train = forward_propagation(row)
        training_predictions.append(Y_train)

        # map target value to output node (e.g. A == node[0])
        target_ltr = X_targets[target_row].tostring()
        target_unit = ltr_to_index[target_ltr]
        # print "target unit:", target_unit
        # record target letter for plotting
        training_letter_actual.append(target_ltr)
        # print training_letter_actual

        # calculate target for each node
        # for node matching letter, t = .9, otherwise t = .1
        output_layer_targets = [.1 for i in range(0, 26)]
        output_layer_targets[target_unit] = .9
        # print "output layer targets", output_layer_targets

        # print "------"
        # print "Y_train:\n", Y_train
        # print "======"
        # compare highest valued output to target unit of .9
        # to see if the neurons have the correct output
        max_value = max(Y_train)
        Y_train_list = Y_train.tolist()
        max_index = Y_train_list.index(max_value)
        # print "\n-------------------"
        # print max_value, max_index, target_unit

        # test to see if max value in neural net output
        # matches the node with the highest target
        # if so, the neural net got the letter correct
        if max_index == target_unit:
            # print "match!"
            correct_train_vote += 1

        # move to next row of input data to use new target
        target_row += 1

    # print "correct train vote", correct_train_vote
    # print "len of training predictions", len(training_predictions)
    training_accuracy = correct_train_vote/len(training_predictions)
    # print "training accuracy", training_accuracy
    # print "-------------------"

    ### Test data ####
    # use forward_propagation(row) to calculate predictions for testing data
    test_predictions = []
    # record values for plotting
    test_letter_vote = []
    test_letter_actual = []
    # reset counter for iterating through letter targets
    target_row = 0
    for row in test_data:
        hidden_layer, Y_test = forward_propagation(row)
        test_predictions.append(Y_test)

        # map target value to output node (e.g. A == node[0])
        target_ltr = X_targets[target_row].tostring()
        target_unit = ltr_to_index[target_ltr]
        # print "target unit:", target_unit
        # record target letter for plotting
        test_letter_actual.append(target_ltr)
        # print test_letter_actual

        # calculate target for each node
        # for node matching letter, t = .9, otherwise t = .1
        output_layer_targets = [.1 for i in range(0, 26)]
        output_layer_targets[target_unit] = .9
        # print "output layer targets", output_layer_targets

        # compare highest valued output to target unit of .9
        # to see if the neurons have the correct output
        max_value = max(Y_test)
        Y_test_list = Y_test.tolist()
        max_index = Y_test_list.index(max_value)
        # print "\n+++++++++++++++++++"
        # print max_value, max_index, target_unit

        # test to see if max value in neural net output
        # matches the node with the highest target
        # if so, the neural net got the letter correct
        if max_index == target_unit:
            # print "match!"
            correct_test_vote += 1

        # move to next row of input data to use new target
        target_row += 1

    # print "correct test vote", correct_test_vote
    # print "len of test predictions", len(test_predictions)
    testing_accuracy = correct_test_vote/len(test_predictions)
    # print "testing accuracy", testing_accuracy
    # print "+++++++++++++++++++"

    return training_accuracy, testing_accuracy


################################################################################################

def plot_results(training_accuracy_list_low_alpha, testing_accuracy_list_low_alpha, training_accuracy_list_high_alpha,
                 testing_accuracy_list_high_alpha):
    """
    Plot results of accuracy computations

    :return:
    """
    # print len(training_accuracy_list)
    # print len(range(1, epochs+1))

    plt.title('Accuracy: Training and Testing, Experiment 3')
    plt.plot(range(1, epochs+1), training_accuracy_list_low_alpha, 'ro', label='Training, alpha=0.05')
    plt.plot(range(1, epochs+1), testing_accuracy_list_low_alpha, 'b^', label='Test, alpha=0.05')
    plt.plot(range(1, epochs+1), training_accuracy_list_high_alpha, 'go', label='Training, alpha=0.6')
    plt.plot(range(1, epochs+1), testing_accuracy_list_high_alpha, 'r^', label='Test, alpha=0.6')
    plt.xticks(np.arange(0, epochs+2), np.arange(0, epochs+2))
    plt.yticks(np.arange(0,1,0.1), np.arange(0,1,0.1))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='upper right', numpoints=1)
    plt.show()



################################################################################################

def letter_to_unit_index(input_ltr):
    """
    Converts a target letter to an indexed number
    corresponding to indices 0-25 in output layer of neural net
    used in computing targets
    :return number mapping to index of output layer for input letter:
    """
    # index in output layer
    index = 0
    for ltr in string.ascii_uppercase:
        index += 1
        if input_ltr == ltr:
            return index

#### alternative to function: dict mapping letters to number (index of unit in output row) ####
ltr_to_index = dict(zip(string.ascii_uppercase, range(0,26)))
# print ltr_to_index['A'] #0
# print ltr_to_index['Z'] #25

################################################################################################

################
# Experiment 3 #
################

######
# main
######
epochs = 25
#train the neural net for <epochs> number of epochs
# using forward and back propagation
# lists for training and testing accuracies over multiple epochs
training_acc_list_low_alpha = []
testing_acc_list_low_alpha = []
training_acc_list_high_alpha = []
testing_acc_list_high_alpha = []
# run training with low momentum
training_acc_list_low_alpha, testing_acc_list_low_alpha = train(epochs, alpha_low)
print "training accuracy (low alpha) list in main:", training_acc_list_low_alpha
print "testing accuracy (low alpha) list in main:", testing_acc_list_low_alpha
# run training with high momentum
training_acc_list_high_alpha, testing_acc_list_high_alpha = train(epochs, alpha_high)
print "training accuracy (high alpha) list in main:", training_acc_list_high_alpha
print "testing accuracy (high alpha) list in main:", testing_acc_list_high_alpha
# plot results of accuracy testing
plot_results(training_acc_list_low_alpha, testing_acc_list_low_alpha, training_acc_list_high_alpha,
             testing_acc_list_high_alpha)
