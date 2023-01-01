import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.engine.training_generator_v1 import predict_generator
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from confusion_matrix import plot_confusion_matrix
from neural_network_model import check_model, nn_model, check_model_new, evaluate_model_, evaluate_new_model, \
    nn_model_new

dataset_dir = "../../img_she_healthy/DATASET/crvx"

m_batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=m_batch_size,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

validation_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=m_batch_size,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

model = nn_model(2)
epoch = 2

history = check_model_new(model, train_generator, validation_generator, epoch, m_batch_size)

prediction_generator = model.predict_generator(train_generator)

predicted_labels = [np.argmax(prediction) for prediction in prediction_generator]

true_labels = [0, 1]
confusion_mat = confusion_matrix(train_generator.classes, predicted_labels)

print(confusion_mat)

evaluate_new_model(history)
