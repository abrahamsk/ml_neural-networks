#!/usr/bin/env python

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16

import sys

increment = 0
run = [1, 2, 3, 4, 5]
# for every letter, run through every perceptron
# and record votes for which letter perceptron returns
for i in run:
    text = "\rTesting instance "+str((increment)+1)+"/"+str(len(run))
    sys.stdout.write(text)

    increment += 1