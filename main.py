# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2021 <<<
# ----------------------------------------------------------------------------------

# region --- Imports ---

import numpy as np
import sys
import time
import contextlib
import msvcrt
import os
from modules.trainer.trainer import Trainer

np.set_printoptions(precision=2)
# endregion


def main():
    """ Modular Reinforcement Learning© main-function
    This function instantiates a new trainer object with the given parameters, connects it to the Unity or OpenAI Gym
    environment and starts the training or testing process.

    Before running this function, please adjust the parameters in the Parameter Choice region as well as the trainer
    configuration file.

    :return: None
    """

    # region  --- Parameter Choice ---

    # Choose between "training", "testing" or "fastTesting
    # If you want to test a trained model or continue learning from a checkpoint enter the model path below
    mode = "training"
    model_path = r""

    # Instantiate a Trainer object with certain choices of parameters and algorithms
    trainer = Trainer()
    interface = 'MLAgentsV18'  # Choose from "MLAgentsV18" (Unity) and "OpenAIGym"
    environment_path = r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\RobotArm\Grabbing\Level0\DoBotEnvironment.exe"  # In case of "OpenAIGym" enter the desired env name here, e.g. "LunarLanderContinuous-v2"

    # Choose from "None", "EpsilonGreedy" and "ICM"
    exploration_algorithm = 'EpsilonGreedy'

    # Choose from "DQN", "DDPG", "TD3", "SAC"
    trainer.select_training_algorithm('SAC')

    # Choose from "None", "LinearCurriculum", "RememberingCurriculum" and "CrossFadeCurriculum"
    trainer.select_curriculum_strategy('LinearCurriculum')

    # Choose from "None" and "SemanticSegmentation"
    preprocessing_algorithm = 'None'
    preprocessing_path = r""  # Enter the path for the preprocessing model

    trainer.save_all_models = True  # Determines if all models or only the actor will be saved during training
    trainer.remove_old_checkpoints = False  # Determines if old model checkpoints will be overwritten

    # endregion

    # region --- Initialization ---

    # Parse the trainer configuration
    trainer.parse_training_parameters("trainer_configs/trainer_config.yaml", "sac")
    # Instantiate the agent
    trainer.async_instantiate_agent(mode, interface, preprocessing_algorithm, exploration_algorithm,
                              environment_path, model_path, preprocessing_path)

    # endregion

    # region --- Training / Testing ---

    # Play new episodes until the training/testing is manually interrupted and receive the latest process information.
    while True:
        if mode == "training":
            mean_episode_length, mean_episode_reward, \
                episodes, training_step, loss, \
                task_properties = next(trainer.training_loop)
        else:
            mean_episode_length, mean_episode_reward, \
                episodes, training_step, loss, \
                task_properties = next(trainer.testing_loop)
        training_duration = trainer.get_elapsed_training_time()

        # Print the latest training or testing stats to the console.
        if episodes - trainer.last_debug_message >= 10:
            trainer.last_debug_message = episodes
            print("Played Episodes: {}, Training Steps: {}, Task Level: {:d}/{:d}\n"
                  "Average Episode Reward: {:.2f}/{:.2f} (for the last {:d} Episodes)\n"
                  "Elapsed Training Time: {:02d}:{:02d}:{:02d}:{:02d}".format(episodes, training_step, task_properties[1], task_properties[0],
                                                                              mean_episode_reward, task_properties[3], task_properties[2],
                                                                              *training_duration))
            print("--------------------------------------------------------------")

    # endregion


def read_input(caption, default, timeout=5):
    start_time = time.time()
    sys.stdout.write('{}{}:'.format(caption, default))
    sys.stdout.flush()
    input_text = ''
    while True:
        if msvcrt.kbhit():
            byte_arr = msvcrt.getche()
            if ord(byte_arr) == 13:  # enter_key
                break
            elif ord(byte_arr) >= 32:  # space_char
                input_text += "".join(map(chr, byte_arr))
        if len(input_text) == 0 and (time.time() - start_time) > timeout:
            break

    print('')  # needed to move to next line
    if len(input_text) > 0:
        return input_text
    else:
        return default


if __name__ == '__main__':
    main()
