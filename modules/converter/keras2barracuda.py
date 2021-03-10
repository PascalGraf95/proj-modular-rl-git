import keras2onnx
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def main():
    path = r"C:\PGraf\Arbeit\RL\GitLab\modular-reinforcement-learning\training\summaries\201217_082156_SAC_VisualRobot0.2\SAC_Actor_Step15044_Reward1.20.h5"
    keras_model = load_model(path)
    keras_model.layers[0]._name = "CameraSensor1"
    keras_model.layers[1]._name = "CameraSensor2"
    keras_model.layers[-2]._name = "action"
    if len(keras_model.output) > 1:
        output = keras_model.output[0]
        keras_model = Model(keras_model.inputs, output)
        keras_model.summary()

    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, target_opset=9, channel_first_inputs=None)
    keras2onnx.save_model(onnx_model, "model.onnx")


if __name__ == '__main__':
    main()