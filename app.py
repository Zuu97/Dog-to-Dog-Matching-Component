import os
import json
import pandas as pd
import numpy as np
from variables import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from inference import InferenceModel
import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.preprocessing.image import img_to_array

import requests
from PIL import Image
from util import *
from flask import Flask
from flask import jsonify
from flask import request
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
'''
        python -W ignore app.py
'''
model = InferenceModel()

app = Flask(__name__)



@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    if len(message) == 2:
        text_pad, label = get_prediction_data(message)
        model.extract_image_features(label)
        n_neighbours = model.predictions(text_pad)
        response = {
            "neighbours": n_neighbours
                    }
        return jsonify(response)
    else:
        return "Please input both Breed and the text content"

if __name__ == "__main__": 
    app.run(debug=True, host=host, port= port, threaded=False, use_reloader=False)
