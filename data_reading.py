import tensorflow as tf
import numpy as np
import warnings



IM_SIZE = 150  # as in original data
BATCH = 32
TRAIN_PATH = 'seg_train'
TEST_PATH = 'seg_test'

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset='training',
    image_size=(IM_SIZE, IM_SIZE),
    seed=100,
    batch_size=BATCH
)


val_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset='validation',
    image_size=(IM_SIZE, IM_SIZE),
    seed=100,
    batch_size=BATCH
)


test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IM_SIZE, IM_SIZE),
    seed=100,
    batch_size=BATCH
)


import matplotlib.pyplot as plt

cls_names = train_data.class_names

train_dataset = train_data.cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_data.cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_data.cache().prefetch(tf.data.AUTOTUNE)