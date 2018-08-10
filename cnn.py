#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 00:43:41 2018

@author: asami234
"""

"""
Created on Wed Aug  8 03:48:54 2018

@author: asami234
"""

''' Part 1  structure the dataset for keras'''

# Part 2 Building the CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout

# Initializing The CNN
classifier =  Sequential()

# Step 1 Convolution , create the convolutional layer, this step automatcially applies all the feature detectors on the input image and then produce the covo layer then apply the relu activation function to ass non linearity
# 32 is the number of feature detectors and (3,3) is the same of each feature detector
# 64 by 64 is the size we want our images to be , tehy will be forced . and 3 is the number of channels in the image . 3 for RGB.
classifier.add(Convolution2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))

# Max Pooling , create the pooling  layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# add another convo layer to improve accuracy 
# no need to put input shape beacuse we wil be convolving the pooled layer in the previos lines, not the input images.Keras will know what is the previous layer.
classifier.add(Convolution2D(32, (3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())


# Full Connection -> Creating the ANN
# hidden layer
classifier.add(Dense(units = 140, activation = 'relu'))
# add dropout regeularization to reduce overfitiing 
classifier.add(Dropout(rate = 0.5))

classifier.add(Dense(units = 140, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))

# output layer 
classifier.add(Dense(units = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# FItting the cnn into images 

# IMage preprocessing -> Image Augemntation 
from keras.preprocessing.image import ImageDataGenerator
# The genarator object which will create the augmanted images for the training set
# notes that these processes below will not be the same for each image.all of them will be different , they will have different scaled values and zoom etccc
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# for the testing set , we will only use the testing generator to scale the test images so they will have random sizs just like the  training images
test_datagen = ImageDataGenerator(rescale=1./255)

# target size is the size expected to be found in the convo layers
# batch size is the number of random images that will go through the ann each iteration
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# test set used to evaluate the model performance
# class_mode is how many categories we have in our dataset . in pur case only 2 dog and cat .Thats why we put 
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
# steps per epoch is how many images will run through the ann per one epoch
# validation steps is the number of images in the test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32)


# Single Prediction 
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
# will allows us to add a third diminsion so we can have 3 channels just like the input shape argument in th convo layer  (64,3,3)
test_image = image.img_to_array(test_image)
# we have to add a 4th dimension that corresponds to the batch , thats what the predict method expects
# axis is the index of the value of the dimension we are adding , it has to be at the first position so we put 0 
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

# get the info of what 0 nd 1 coorspond to
training_set.class_indices

if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'