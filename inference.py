import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.neighbors import NearestNeighbors
from util import load_rcn_data, load_inference_data, get_prediction_data, load_labeled_data, rescale_imgs
from variables import *

from rcn import DoggyRCN
from tflite_converter import KerasToTFConversion
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class InferenceModel(object):
    def __init__(self):
        image_labels, inference_images, image_urls = load_inference_data()
        self.inference_images = inference_images
        self.image_labels = image_labels
        self.image_urls = image_urls
        rcn_inference = KerasToTFConversion()
        cnn_inference = KerasToTFConversion()

        if (not os.path.exists(cnn_converter_path)) or (not os.path.exists(rcn_converter_path)):
            self.rcn_model_obj = DoggyRCN()
            self.rcn_model_obj.run()

            if not os.path.exists(cnn_converter_path):
                ccn_model = self.rcn_model_obj.cnn_encoder
                cnn_inference.TFconverter(ccn_model, cnn_weights, cnn_converter_path)
                print(" CNN keras model Converted to TensorflowLite")

            if not os.path.exists(rcn_converter_path):
                rcn_model = self.rcn_model_obj.model
                rcn_inference.TFconverter(rcn_model, rcn_weights, rcn_converter_path)
                print(" RCN keras model Converted to TensorflowLite")

        cnn_inference.TFinterpreter(cnn_converter_path)
        rcn_inference.TFinterpreter(rcn_converter_path)

        self.rcn_inference = rcn_inference
        self.cnn_inference = cnn_inference
        
    def extract_image_features(self, label):
        self.image_labels, self.inference_images, self.image_urls = load_labeled_data(
                                                                                self.image_labels, 
                                                                                self.inference_images, 
                                                                                self.image_urls,
                                                                                label)
        if not os.path.exists(n_neighbour_weights.format(label)):
            self.test_features = np.array(
                            [self.cnn_inference.Inference(img) for img in self.inference_images]
                                        )
            self.test_features = self.test_features.reshape(self.test_features.shape[0],-1)
            self.neighbor = NearestNeighbors(
                                        n_neighbors = n_neighbour
                                        )
            self.neighbor.fit(self.test_features)
            with open(n_neighbour_weights.format(label), 'wb') as file:
                pickle.dump(self.neighbor, file)
        else:
            with open(n_neighbour_weights.format(label), 'rb') as file:
                self.neighbor = pickle.load(file)

    def extract_text_features(self, text_pad):
        return self.rcn_inference.Inference(text_pad)

    def predictions(self, text_pad, show_fig=False):
        n_neighbours = {}
        fig=plt.figure(figsize=(8, 8))
        text_pad = self.extract_text_features(text_pad)
        text_pad = text_pad.reshape(1, -1)
        result = self.neighbor.kneighbors(text_pad)[1].squeeze()
        for i in range(n_neighbour):
            neighbour_img_id = result[i]
            img = self.inference_images[neighbour_img_id]
            url = self.image_urls[neighbour_img_id]
            img = rescale_imgs(img)
            fig.add_subplot(1, 3, i+1)
            plt.title('Neighbour {}'.format(i+1))
            plt.imshow((img * 255).astype('uint8'))
            n_neighbours["Neighbour {}".format(i+1)] =  "{}".format(url)
        if show_fig:
            plt.show()
        return n_neighbours

# message = {
#     "text" : "I would like to arrange a playdate for my female small size Maltese puppy it is very playful,active and have a good behaviour with other pets and behave well with strangers love go for walks. we live in kalutara.",
#     "label" : "shih tzu"
#     }


# model = InferenceModel()
# text_pad, label = get_prediction_data(message)
# model.extract_image_features(label)
# model.predictions(text_pad)