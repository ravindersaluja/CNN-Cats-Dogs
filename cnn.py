#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:46:01 2020

@author: ravindersaluja
"""

# Importing Libs
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classfier = Sequential()
# input_shape (Reducing the shape because using CPU and not GPU, so (64,64,3), 3 being the channels - RGB)
# Channels last because using tensorflow backend.
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3)), activation='relu')
