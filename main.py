# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2022 <<<
# ----------------------------------------------------------------------------------

# region --- Imports ---
import numpy as np
from modules.trainer import Trainer
np.set_printoptions(precision=2)
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
    model_path = None

    # Instantiate a Trainer object with certain choices of parameters and algorithms
    trainer = Trainer()
    interface = 'MLAgentsV18'  # Choose from "MLAgentsV18" (Unity) and "OpenAIGym"
    # If you want to run multiple Unity actors in parallel you need to specify the path to the Environment '.exe' here.
    # In case of "OpenAIGym" enter the desired env name here instead, e.g. "LunarLanderContinuous-v2"
    environment_path = None

    # - Training Algorithm -
    # This is the core learning algorithm behind the RL Agent. While Deep Q-Learning / Deep Q Networks (DQN) presumably
    # is the most famous algorithm it can only act in environments with discrete action space. The three others
    # in their current implementation only support continuous action spaces. Of those three Soft Actor-Critic (SAC)
    # is the most recent and preferred option.
    # Choose from "DQN", "DDPG", "TD3", "SAC"
    trainer.select_training_algorithm('SAC')

    # - Exploration Algorithm -
    # The exploration algorithm helps the RL Agent to explore the environment by occasionally choosing suboptimal
    # actions or giving reward bonuses to unseen states instead of exploiting the current knowledge.
    # Choose from "None", "EpsilonGreedy", "ICM" and "RND"
    exploration_algorithm = "EpsilonGreedy"

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
    preprocessing_path = r"C:\PGraf\Arbeit\RL\SemanticSegmentation\vae\models\210809_101443_VAE_encoder_235"

    # - Misc -
    trainer.save_all_models = True  # Determines if all models or only the actor will be saved during training
    trainer.remove_old_checkpoints = False  # Determines if old model checkpoints will be overwritten

    # endregion

    # region --- Initialization ---

    # Parse the trainer configuration (make sure to select the right key)
    trainer.parse_training_parameters("trainer_configs/trainer_config.yaml", "sac")
    # Instantiate the agent which consists of a learner and one or multiple actors
    trainer.async_instantiate_agent(mode, interface, preprocessing_algorithm, exploration_algorithm,
                                    environment_path, model_path, preprocessing_path)
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
