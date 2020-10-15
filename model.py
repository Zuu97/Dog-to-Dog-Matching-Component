import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from util import *
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DogMatchingComponent(object):
    def __init__(self):
        if not (os.path.exists(model_architecture)  and os.path.exists(model_weights)):
            train_generator, validation_generator = image_data_generator()
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.train_step = self.train_generator.samples // batch_size
            self.validation_step = self.validation_generator.samples // batch_size

    def model_conversion(self): #MobileNet is not build through sequential API, so we need to convert it to sequential
        mobilenet_functional = tf.keras.applications.MobileNet()
        model = Sequential()
        for layer in mobilenet_functional.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            layer.trainable = False
            model.add(layer)
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(dense_4, activation='relu'))
        model.add(Dense(dense_4, activation='relu'))
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
                          steps_per_epoch= self.train_step,
                          validation_data = self.validation_generator,
                          validation_steps = self.validation_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        print("Mobile Net TF Model Saving !")
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_weights)

        self.model.compile(
                           loss='categorical_crossentropy', 
                           optimizer='Adam', 
                           metrics=['accuracy']
                           )
        print("Mobile Net TF Model Loaded !")


    def run_MobileNet(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            t1 = time.time()
            self.model_conversion()
            t2 = time.time()
            print(t2-t1)
            self.train()
            t3 = time.time()
            self.save_model()
            print(t3-t2)


model = DogMatchingComponent()
model.run_MobileNet()