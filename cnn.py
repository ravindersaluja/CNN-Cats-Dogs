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
from keras.preprocessing.image import ImageDataGenerator

# Initializing
classifier = Sequential()


# Adding the Convolutional layer and Activation fn
# input_shape (Reducing the shape because using CPU and not GPU, so (64,64,3), 3 being the channels - RGB)
# Channels last because using tensorflow backend.
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

# Adding the Pooling layer
# Adding the max pooling layer to decrease the number of nodes in the future fully connected layers
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flatenning
classifier.add(Flatten())

# Full connection
# units = hidden nodes in the hidden layer 128-not too big not too small and taking pow of 2.
# Input layer
classifier.add(Dense(units=128, activation = 'relu'))
# Output layer
classifier.add(Dense(units=1, activation = 'sigmoid'))
# classifier.summary()

# Compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image augmentation - image preprocessing to avoid overfitting-rotates/flips etc the image and creates more batches
# of it if the dataset is too small and thus reducing overfitting
# Using ImageDataGenerator class for the same.

with open('pathFile.txt', 'r') as f:
    l = f.read().splitlines()
    trainingPath = l[0]+'training_set'
    testingPath = l[0]+'test_set'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        trainingPath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
    testingPath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)



