from tensorflow.keras.models import load_model
from datetime import datetime
from keras import Model
import numpy as np

def main(path, iterations=50):
    test_model = load_model(path)
    input_shape = test_model.input_shape
    random_input = np.random.random((iterations, *input_shape[1:]))
    print("Random Input Batch Shape:", random_input.shape)

    start_time = datetime.now()
    for i in range(50):
        test_model.predict(np.expand_dims(random_input[i], axis=0))
    average_prediction_time = (datetime.now() - start_time).microseconds/1000/iterations

    print("Average Network Prediction Time:", average_prediction_time, "ms")


if __name__ == '__main__':
    main(r"C:\PGraf\Arbeit\RL\GitLab\modular-reinforcement-learning\training\pretrained_weights\VectorRobot0.8\SAC_Actor_Step163892_Reward0.89.h5")