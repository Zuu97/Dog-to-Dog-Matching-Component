import tensorflow as tf
import numpy as np
import os
import numpy as np
import pandas as pd
import cv2 as cv

from variables import*

def get_class_names():
    return os.listdir(train_dir)

def image_data_generator():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale = rescale,
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split
                                    )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = get_class_names(),
                                    shuffle = True)

    validation_generator = test_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = get_class_names(),
                                    shuffle = True)

    return train_generator, validation_generator
