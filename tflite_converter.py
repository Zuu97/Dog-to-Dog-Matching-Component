import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
import pathlib
logging.getLogger('tensorflow').disabled = True
from variables import*
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class KerasToTFConversion(object):
    def TFconverter(self, feature_model, model_weights, model_converter):
        # if model_weights == rcn_weights:
        #     converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_weights)
        # else:
        converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
                                tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS
                                ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(model_converter)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self, model_converter):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_converter)
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