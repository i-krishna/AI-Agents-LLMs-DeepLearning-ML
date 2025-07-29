# Building a TensorFlow Deep Learning Model from Scratch to Deployment

# building, training,  evaluating, deploying as a web service using Flask
# of image classification model (using the CIFAR-10 dataset), 

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from flask import Flask, request, jsonify

# Load CIFAR-10 dataset
# CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Verify the data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()