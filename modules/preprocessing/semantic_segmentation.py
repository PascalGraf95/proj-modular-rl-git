from .preprocessing_blueprint import PreprocessingAlgorithm
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf


class SemanticSegmentation(PreprocessingAlgorithm):
    Name = "SemanticSegmentation"

    def __init__(self, model_path=""):
        with tf.device('/cpu:0'):
            self.segmentation_model = load_model(model_path)

    def preprocess_observations(self, decision_steps, terminal_steps):
        with tf.device('/cpu:0'):
            if len(decision_steps):
                for idx, o in enumerate(decision_steps.obs):
                    if len(o.shape) == 4:
                        decision_steps.obs[idx] = self.segmentation_model.predict(o)[0]
            if len(terminal_steps):
                for idx, o in enumerate(terminal_steps.obs):
                    if len(o.shape) == 4:
                        terminal_steps.obs[idx] = self.segmentation_model.predict(o)[0]
        return decision_steps, terminal_steps

    def get_output_shapes(self, environment_configuration):
        with tf.device('/cpu:0'):
            output_shapes = []
            for obs_shape in environment_configuration["ObservationShapes"]:
                if len(obs_shape) == 3:
                    output_shapes.append(self.segmentation_model.output_shape[0][1:])
                else:
                    output_shapes.append(obs_shape)
            return output_shapes

