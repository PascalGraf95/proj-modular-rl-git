# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2022 <<<
# ----------------------------------------------------------------------------------

# region --- Imports ---
import numpy as np
from modules.trainer import Trainer
np.set_printoptions(precision=2)

import random
import tensorflow as tf
random.seed(997)
np.random.seed(997)
tf.random.set_seed(997)
# endregion


def main():
    """ Modular Reinforcement Learning© main-function
    This function defines the parameters to instantiate a new trainer object which then creates one or multiple actors
    and learners to connect to a Unity or OpenAI Gym environment. After that the main-function starts the training
    or testing procedure.

    Before running this function, please adjust the parameters in the Parameter Choice region as well as the trainer
    configuration under the respective key (./trainer_configs/trainer_config.yaml).

    :return: None
    """
    # region  --- Parameter Choice ---

    # Choose between "training", "testing" or "fastTesting"
    # If you want to test a trained model or continue learning from a checkpoint enter the model path below
    mode = "training"
    #model_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\3_Software\proj-modular-reinforcement-learning\training\summaries\220831_084334_SAC_MountainCar_R2D2"
    #model_path =  r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\3_Software\proj-modular-reinforcement-learning\training\summaries\220831_151730_SAC_MountainCar_Agent57HP"
    #model_path =  r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\3_Software\proj-modular-reinforcement-learning\training\summaries\220831_211254_SAC_MountainCarV0_Agent57rHP"
    model_path = None #r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\3_Software\proj-modular-reinforcement-learning\training\summaries\220909_094200_SAC_MountainCarv0_Agent57noRewards"

    # Instantiate a Trainer object with certain choices of parameters and algorithms
    trainer = Trainer()
    trainer.interface = 'MLAgentsV18'  # Choose from "MLAgentsV18" (Unity) and "OpenAIGym"
    # If you want to run multiple Unity actors in parallel you need to specify the path to the Environment '.exe' here.
    # In case of "OpenAIGym" enter the desired env name here instead, e.g. "LunarLanderContinuous-v2".
    # If you want a CQL agent to learn from demonstrations, an environment can be used to evaluate the model on a
    # regular basis. Please provide a path or type None to connect directly to the Unity Editor. Otherwise, type
    # 'NoEnv' to proceed without evaluation.
    environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\3DBall\Skiing.exe"
    #environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\minimHallway\Hallway.exe"
    #environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\Crawler\Crawler.exe"
    #environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\Walker\Env.exe"
    #environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\HallwayDQN\Env.exe"
    #environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\BasicDQN\Env.exe"
    #environment_path = r"C:\Users\Martin\Desktop\HS Heilbronn\Master\Masterthesis\4_Unity\_built\BasicDQNHard\Env.exe"

    #environment_path = r"CartPole-v1"
    #environment_path = r"MountainCar-v0"
    #environment_path = r"MountainCarContinuous-v0"
    #environment_path = r"Acrobot-v1"
    #environment_path = r"BipedalWalker-v3"
    #environment_path = r"Reacher-v4"
    #environment_path = r"LunarLander-v2"
    #environment_path = r"Reacher-v4"
    #environment_path = r"HalfCheetah-v4"
    #environment_path = r"Humanoid-v4"

    #environment_path = r"maze-random-10x10-plus-v0"
    #environment_path = r"InvertedPendulum-v4"

    # - Training Algorithm -
    # This is the core learning algorithm behind the RL Agent. While Deep Q-Learning / Deep Q Networks (DQN) presumably
    # is the most famous algorithm it can only act in environments with discrete action space. The three others
    # in their current implementation only support continuous action spaces. Of those three Soft Actor-Critic (SAC)
    # is the most recent and preferred option.
    # Choose from "DQN", "DDPG", "TD3", "SAC", "CQL"
    training_algorithm = 'SAC'
    trainer.select_training_algorithm(training_algorithm)
    # In case you want to train the agent offline via CQL please provide the path for demonstrations.
    demonstration_path = None  #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-robot-arm-environment\DemoSinglePendulum"

    # - Exploration Algorithm -
    # The exploration algorithm helps the RL Agent to explore the environment by occasionally choosing suboptimal
    # actions or giving reward bonuses to unseen states instead of exploiting the current knowledge.
    # Choose from "None", "EpsilonGreedy", "ICM", "RND", "ENM", "NGU", "ECR", "NGUr"
    exploration_algorithm = "EpsilonGreedy"

    # - Meta Learning Algorithm -
    # The meta learning algorithm helps the RL Agent to learn the most efficient way of learning.
    # Choose from "None", "MetaController"
    meta_learning_algorithm = "None"

    # - Curriculum Strategy -
    # Just like humans, a RL Agent learns best by steadily increasing the difficulty of the given task. Thus, for
    # Unity Environments an option to choose a task level has been implemented. Notice, however, that in order to work
    # the Unity environment needs special preparation (like a SideChannel reading the messages).
    # Choose from None, "LinearCurriculum" ("RememberingCurriculum" and "CrossFadeCurriculum" are currently disabled)
    curriculum_strategy = None
    trainer.select_curriculum_strategy(curriculum_strategy)

    # - Preprocessing Algorithm -
    # In some cases it can be useful to present not just the raw data from the environment to the RL Agent but to
    # apply some preprocessing first. In case of Semantic Segmentation, a previously trained Variational Autoencoder
    # takes the image input and transforms it into a much more compact representation leading to faster convergence.
    # Choose from "None" and "SemanticSegmentation"
    preprocessing_algorithm = 'None'
    # Enter the path for the preprocessing model if needed
    preprocessing_path = None  #r"C:\PGraf\Arbeit\RL\SemanticSegmentation\vae\models\210809_101443_VAE_encoder_235"

    # - Misc -
    trainer.save_all_models = True  # Determines if all models or only the actor will be saved during training
    trainer.remove_old_checkpoints = True  # Determines if old model checkpoints will be overwritten
    # endregion

    # region --- Initialization ---

    # Parse the trainer configuration (make sure to select the right key)
    trainer.parse_training_parameters("trainer_configs/trainer_config.yaml", training_algorithm.lower())
    # Instantiate the agent which consists of a learner and one or multiple actors
    trainer.async_instantiate_agent(mode, preprocessing_algorithm, exploration_algorithm, meta_learning_algorithm,
                                    environment_path, model_path, preprocessing_path, demonstration_path)
    # If you are trying to understand this project, the next place to continue exploring it would be the trainer file
    # in the respective directory (./modules/trainer.py)

    # endregion

    # region --- Training / Testing ---
    # Play new episodes until the training/testing is manually interrupted
    if mode == "training":
        if curriculum_strategy and curriculum_strategy != "None":
            trainer.async_training_loop_curriculum()
        else:
            trainer.async_training_loop()
    else:
        trainer.async_testing_loop()
    # endregion


if __name__ == '__main__':
    main()
