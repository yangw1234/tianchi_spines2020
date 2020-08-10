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

import spines

from zoo import init_nncontext

sc = init_nncontext()

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

import numpy as np
train_data = np.load("../../data/train.npy", allow_pickle=True)
val_data = np.load("../../data/val.npy", allow_pickle=True)
keys = ["images", "image_meta", "rpn_match", "rpn_bbox", "gt_class_ids", "gt_boxes_nd", "gt_masks_nd"]
x = [train_data.item()[key] for key in keys]
y = []

val_x = [val_data.item()[key] for key in keys]
val_y = []

from zoo.orca.data.shard import XShards

train_x_shards = XShards.partition({"x": tuple(x), "y": tuple(y)})
val_x_shards = XShards.partition({"x": tuple(val_x), "y": tuple(val_y)})

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.compile(config.LEARNING_RATE, config.LEARNING_MOMENTUM)

import tensorflow as tf
optimizer = tf.keras.optimizers.SGD(
        lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM,
        clipnorm=config.GRADIENT_CLIP_NORM)
    
model.keras_model.compile(optimizer=optimizer, loss=[None] * 14)

loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

metrics = {}

# Add metrics for losses
for name in loss_names:
    if name in model.keras_model.metrics_names:
        continue
    layer = model.keras_model.get_layer(name)
    model.keras_model.metrics_names.append(name)
    loss = (
        tf.reduce_mean(layer.output, keepdims=True)
        * model.config.LOSS_WEIGHTS.get(name, 1.))
    metrics[name] = loss
metrics["total_loss"] = model.keras_model.total_loss

from zoo.orca.learn.tf.estimator import Estimator

estimator = Estimator.from_keras(model.keras_model, metrics=metrics, model_dir="./logs")
estimator.fit(data=train_x_shards,
              epochs=30,
              batch_size=32*2,
              validation_data=val_x_shards,
              hard_code_batch_size=True,
              session_config=tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=24))
