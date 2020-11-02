import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.neighbors import NearestNeighbors
from util import load_final_data
from variables import *

from cnn import DoggyCNN
from final_model import DogToDogMatchingComponent

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class KerasToTFConversion(object):
    def __init__(self, feature_model, model_converter, model_weights):
        self.feature_model = feature_model
        self.model_converter = model_converter
        self.model_weights = model_weights

    def TFconverter(self):
        if self.model_converter == final_model_converter:
            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(self.model_weights)
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(self.model_converter)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=self.model_converter)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Inference(self, inputs):
        input_shape = self.input_details[0]['shape']
        input_data = np.expand_dims(inputs, axis=0).astype(np.float32)
        assert np.array_equal(input_shape, input_data.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
        
class InferenceModel(object):
    def __init__(self):
        Xtrain_pad, Xtest_pad, Imgtrain, Imgtest, tokenizer = load_final_data()
        self.Xtrain_pad = Xtrain_pad
        self.Xtest_pad = Xtest_pad
        self.Imgtrain = Imgtrain
        self.Imgtest = Imgtest
        self.tokenizer = tokenizer

        print(" Train image Shape : {}".format(Imgtrain.shape))
        print(" Test  image Shape : {}".format(Imgtest.shape))
        print(" Train review Shape: {}".format(Xtrain_pad.shape))
        print(" Test review Shape : {}".format(Xtest_pad.shape))

        self.final_model_obj = DogToDogMatchingComponent()
        self.cnn_model_obj = DoggyCNN()
        self.final_model_obj.run()
        self.cnn_model_obj.run()

        final_model = self.final_model_obj.model
        cnn_model = self.cnn_model_obj.model

        final_model_conversion = KerasToTFConversion(final_model, final_model_converter, final_model_weights)
        cnn_model_conversion = KerasToTFConversion(cnn_model, cnn_model_converter, cnn_weights)
        final_model_conversion.TFconverter()
        cnn_model_conversion.TFconverter()

InferenceModel()