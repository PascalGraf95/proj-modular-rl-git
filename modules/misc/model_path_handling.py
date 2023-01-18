import os
import numpy as np


def create_model_dictionary_from_path(model_path):
    """
    Given a model path create a dictionary with a unique key for each network model in that path where the value
    for each model again consists of a dictionary with keys "TrainingStep", "Rating", "Reward" and "ModelPaths".
    The returned dictionary can be utilized to create a tournament schedule and to load specific models selected
    by certain attributes.
    :param model_path: Path containing either one specific or multiple models in subdirectories.
    :return:
    """
    # If the given path does not exist return an empty model dictionary.
    if not model_path or not os.path.isdir(model_path):
        return {}
    # Determine if the provided model path contains multiple training folders of which each might contain
    # multiple model checkpoints itself. This is decided by the fact if the folders in the path contain a date at
    # the beginning (which is ensured for all new training sessions with this framework).
    training_session_paths = []
    for file in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, file)):
            if file[:6].isdigit():
                training_session_paths.append(os.path.join(model_path, file))
    # In case there is any file path in this list, we expect it to be in the top folder containing one or multiple
    # training sessions of which each might hold one or multiple trained models. Otherwise, the provided path
    # already points to one specific training session.
    if len(training_session_paths) == 0 and model_path:
        training_session_paths.append(model_path)

    # Create a dictionary to store all provided models by a unique identifier, along with their paths, rating, etc.
    model_dictionary = {}
    # For each training session folder search for all network checkpoints it contains
    for path in training_session_paths:
        for file in os.listdir(path):
            # There are at most two different ways of how models can be stored. Either a model consists of a
            # folder with multiple files in it, or it is stored as .h5 file.
            # Check if the respective folder/file is a model checkpoint.
            # This is decided by the fact if the folder/file contains the keywords "Step" and "Reward"
            # (which is ensured for all checkpoints created with this framework).
            if "Step" in file and "Reward" in file:
                # To retrieve the training step of the checkpoint, split the file string.
                training_step = [f for f in file.split("_") if "Step" in f][0]
                training_step = training_step.replace("Step", "")

                # To retrieve the training reward of the checkpoint, split the file string.
                training_reward = [f for f in file.split("_") if "Reward" in f][0]
                training_reward = training_reward.replace("Reward", "")
                # In case of a .h5 file, remove the file ending.
                training_reward = training_reward.replace(".h5", "")

                # The unique identifier (key) is the training session name along with the training step
                key = path.split("\\")[-1] + "_" + training_step

                # If the key is not already in the dictionary, append it. Otherwise, just add the current model
                # to its model paths. This case is possible due to the fact, that one training checkpoint might
                # consist of multiple models, e.g., two critics and one actor for SAC.
                if key not in model_dictionary:
                    model_dictionary[key] = {"TrainingStep": int(training_step),
                                             "Rating": 0,
                                             "Reward": float(training_reward),
                                             "ModelPaths": [os.path.join(path, file)]}
                else:
                    model_dictionary[key]["ModelPaths"].append(os.path.join(path, file))
    return model_dictionary


def get_model_key_from_dictionary(model_dictionary, mode="latest"):
    # If the model dictionary is empty return None
    if len(model_dictionary) == 0:
        return None
    # Get the most recent model by date and time.
    if mode == "latest":
        sorted_key_list = sorted(list(model_dictionary.keys()))
        return sorted_key_list[-1]
    # Get a random model.
    elif mode == "random":
        key_list = list(model_dictionary.keys())
        return np.random.choice(key_list, 1)[0]
    # Get the model with the highest reward (might not be representative for different trainings).
    elif mode == "best":
        sorted_key_list = [key for key, val in sorted(model_dictionary.items(), key=lambda item: item[1]['Reward'])]
        return sorted_key_list[-1]
    # Get the model with the highest elo. In case there is no elo rating this will return a random model.
    elif mode == "elo":
        sorted_key_list = [key for key, val in sorted(model_dictionary.items(), key=lambda item: item[1]['Rating'])]
        return sorted_key_list[-1]
    # Otherwise, we expect the mode variable to contain the key itself (used for matches between two distinct models
    # for elo rating.
    else:
        return mode
