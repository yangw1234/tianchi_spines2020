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
EXECUTOR_NUM = 2
EXECUTOR_THREADS = 24
DRIVER_MEMORY = "10G"
EXECUTOR_MEMORY = "80g"

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")
LIB_TF = os.path.join(ROOT_DIR, "tf_libs")
assert os.path.exists(os.path.join(LIB_TF, "libtensorflow_framework-zoo.so")), f"Cannot found tf_libs, please download and extract it to {LIB_TF}"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data/train.npy")
assert os.path.exists(TRAIN_DATA_PATH), f"Cannot find train data, please use generate_ndarray_data.py to generate train data to {TRAIN_DATA_PATH}"
VAL_DATA_PATH = os.path.join(ROOT_DIR, "data/val.npy")
assert os.path.exists(TRAIN_DATA_PATH), f"Cannot find val data, please use generate_ndarray_data.py to generate train data to {VAL_DATA_PATH}"
sys.path.append(ROOT_DIR)

os.environ["KMP_BLOCKTIME"]="1"
os.environ["KMP_AFFINITY"]="disabled"
os.environ["OMP_NUM_THREADS"]=str(EXECUTOR_THREADS)

from zoo import init_spark_standalone, stop_spark_standalone

sc = init_spark_standalone(
    num_executors=EXECUTOR_NUM,
    executor_cores=1, # this parameter will control the number of models in each executor, so we set 1 here. OMP_NUM_THREADS will control the actual number threads used.
    driver_memory=DRIVER_MEMORY,
    executor_memory=EXECUTOR_MEMORY,
    conf={"spark.driver.extraJavaOptions": f"-Djava.library.path={LIB_TF}",
          "spark.executor.extraJavaOptions": f"-Djava.library.path={LIB_TF}"}
)

# Import Mask RCNN
import mrcnn.model as modellib

config = spines.SpinesConfig()
config.display()

import numpy as np
train_data = np.load(TRAIN_DATA_PATH, allow_pickle=True)
val_data = np.load(VAL_DATA_PATH, allow_pickle=True)
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
              batch_size=config.BATCH_SIZE * EXECUTOR_NUM,
              validation_data=val_x_shards,
              hard_code_batch_size=True,
              session_config=tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=EXECUTOR_THREADS))


stop_spark_standalone()