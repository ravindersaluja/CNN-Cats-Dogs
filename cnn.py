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

# Initializing
classifier = Sequential()


# Adding the Convolutional layer and Activation fn
# input_shape (Reducing the shape because using CPU and not GPU, so (64,64,3), 3 being the channels - RGB)
# Channels last because using tensorflow backend.
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

# Adding the Pooling layer
# Adding the max pooling layer to decrease the number of nodes in the future fully connected layers
classfier.add(MaxPooling2D(pool_size=(2,2)))

# Flatenning
classifier.add(Flatten())
