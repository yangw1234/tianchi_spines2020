#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:


import os
import sys
import numpy as np
# from zoo.tfpark import TFDataset

import spines

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model_path = os.path.join(MODEL_DIR, "mask_rcnn_spines.h5")
# Local path to trained weights file


config = spines.SpinesConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.compile(config.LEARNING_RATE, config.LEARNING_MOMENTUM)

import tensorflow as tf
optimizer = tf.keras.optimizers.SGD(
        lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM,
        clipnorm=config.GRADIENT_CLIP_NORM)
    
model.keras_model.compile(optimizer=optimizer, loss=[None] * 14)

data_size = 100

import numpy as np
dtypes_i = (np.float32, np.int64, np.int32, np.float64, np.int32, np.int32, np.bool)
shapes_i = ((data_size, 512, 512, 1), (data_size, 21), (data_size, 65472, 1), (data_size, 256, 4), (data_size, 100), (data_size, 100, 4), (data_size, 56, 56, 100))

x = [np.zeros(shape=shape, dtype=dtype) for shape, dtype in zip(shapes_i, dtypes_i)]
y = [np.zeros(shape=(data_size))] * 14
y = []

model.keras_model.fit(x, y)


