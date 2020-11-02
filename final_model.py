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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from util import load_final_data
from variables import *

from cnn import DoggyCNN
from rnn import DoggyRNN

class DogToDogMatchingComponent(object):
    def __init__(self):
        if not os.path.exists(final_model_weights):
            Xtrain_pad, Xtest_pad, Imgtrain, Imgval, tokenizer = load_final_data()
            self.Xtrain_pad = Xtrain_pad
            self.Xtest_pad = Xtest_pad
            self.Imgtrain = Imgtrain
            self.Imgtest = Imgtest
            self.tokenizer = tokenizer

            print(" Train image Shape : {}".format(Imgtrain.shape))
            print(" Test  image Shape : {}".format(Imgtest.shape))
            print(" Train review Shape: {}".format(Xtrain_pad.shape))
            print(" Test review Shape : {}".format(Xtest_pad.shape))
            
            self.rnn_model = DoggyRNN()
            self.cnn_model = DoggyCNN()
            self.rnn_model.run()
            self.cnn_model.run()

    def build_rnn_encoder(self):
        self.rnn_lstm= self.rnn_model.model
        inputs = self.rnn_lstm.input
        outputs = self.rnn_lstm.layers[-3].output
        self.rnn_encoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )
                            
    def build_cnn_encoder(self):
        self.cnn_mobilenet = self.cnn_model.model
        inputs = self.cnn_mobilenet.input
        outputs = self.cnn_mobilenet.layers[-2].output
        self.cnn_encoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )

    def image_extraction(self):
        self.Ytrain = self.cnn_encoder.predict(self.Imgtrain)
        self.Ytest  = self.cnn_encoder.predict(self.Imgtest)

    def dogMatcher(self):
        self.build_rnn_encoder()
        self.build_cnn_encoder()
        self.image_extraction()

        input_rnn = self.rnn_encoder.input
        output_cnn = self.cnn_encoder.output

        output_rnn = self.rnn_encoder(input_rnn)
        x = Dense(dense_1_final, activation='relu')(output_rnn)
        x = Dense(dense_1_final, activation='relu')(x)
        out_cnn = Dense(dense_2_final, activation='relu')(x)

        self.model = Model(
                        inputs = input_rnn,
                        outputs = out_cnn,
                        name = 'RCN'
                        )
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='mse'
                          )
        self.model.fit(
                        self.Xtrain_pad,
                        self.Ytrain,
                        validation_data=[self.Xtest_pad, self.Ytest],
                        epochs=epochs_final,
                        verbose=verbose
                        )

    def save_model(self):
        self.model.save(final_model_weights)
        print(" RCN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(final_model_weights)
        print(" RCN Model Loaded")


    def run(self):
        if os.path.exists(final_model_weights):
            self.load_model()
        else:
            self.dogMatcher()
            self.train()
            self.save_model()

final_model = DogToDogMatchingComponent()
final_model.run()

