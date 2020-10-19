import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, BatchNormalization
from tensorflow.keras.optimizer import Adam
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from util import load_image_data
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DoggyCNN(object):
    def __init__(self):
        if not (os.path.exists(cnn_architecture)  and os.path.exists(cnn_weights)):
            train_generator, validation_generator = load_image_data()
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.train_step = self.train_generator.samples // batch_size_cnn
            self.validation_step = self.validation_generator.samples // batch_size_cnn

    def model_conversion(self): #MobileNet is not build through sequential API, so we need to convert it to sequential
        mobilenet_functional = tf.keras.applications.MobileNet()
        model = Sequential()
        for layer in mobilenet_functional.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            layer.trainable = False
            model.add(layer)
        model.add(Dense(dense_1_cnn, activation='relu'))
        model.add(Dense(dense_1_cnn, activation='relu'))
        model.add(Dense(dense_2_cnn, activation='relu'))
        model.add(Dense(dense_2_cnn, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(dense_3_cnn, activation='relu'))
        model.add(Dense(dense_3_cnn, activation='relu'))
        model.add(Dense(dense_3_cnn, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()
        self.model = model

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch = self.train_step,
                          validation_data = self.validation_generator,
                          validation_steps = self.validation_step,
                          epochs=epochs_cnn,
                          verbose=verbose
                        )

    def save_model(self):
        print("Mobile Net TF Model Saving !")
        model_json = self.model.to_json()
        with open(cnn_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(cnn_weights)

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        json_file = open(cnn_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(cnn_weights)

        self.model.compile(
                           loss='categorical_crossentropy', 
                           optimizer='Adam', 
                           metrics=['accuracy']
                           )
        print("Mobile Net TF Model Loaded !")


    def run(self):
        if os.path.exists(cnn_weights):
            self.load_model()
        else:
            self.model_conversion()
            self.train()
            self.save_model()

model = DoggyCNN()
model.run()
