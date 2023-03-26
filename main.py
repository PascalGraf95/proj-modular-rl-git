# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2023 <<<
# ----------------------------------------------------------------------------------

# region --- Imports ---
import numpy as np
from modules.trainer import Trainer
from modules.misc.cmd_arg_parse import CmdArgParse
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
    # Parse training / testing parameters from command line. If not provided default parameters will be used.
    parser = CmdArgParse()
    args = parser.return_args()

    # Choose between "training", "testing" or "fastTesting"
    # If you want to test a trained model or continue learning from a checkpoint enter the model path via command line
    mode = args.mode
    model_path = args.model_path
    # If defined this path defines the clone's weights for self-play training/testing. Otherwise, model_path will be
    # used.
    clone_path = args.clone_path

    # Instantiate a Trainer object with certain choices of parameters and algorithms
    trainer = Trainer()
    trainer.interface = args.trainer_interface  # Choose from "MLAgentsV18" (Unity) and "OpenAIGym"
    # If you want to run multiple Unity actors in parallel you need to specify the path to the Environment '.exe' here.
    # In case of "OpenAIGym" enter the desired env name here instead, e.g. "LunarLanderContinuous-v2".
    # If you want a CQL agent to learn from demonstrations, an environment can be used to evaluate the model on a
    # regular basis. Please provide a path or type None to connect directly to the Unity Editor. Otherwise, type
    # 'NoEnv' to proceed without evaluation.
    environment_path = args.env_path

    # - Training Algorithm -
    # This is the core learning algorithm behind the RL Agent. While Deep Q-Learning / Deep Q Networks (DQN) presumably
    # is the most famous algorithm it can only act in environments with discrete action space. The three others
    # in their current implementation only support continuous action spaces. Of those three Soft Actor-Critic (SAC)
    # is the most recent and preferred option.
    # Choose from "DQN", "SAC", "CQL"
    trainer.select_training_algorithm(args.train_algorithm)
    # In case you want to train the agent offline via CQL please provide the path for demonstrations.
    demonstration_path = args.demo_path 

    # - Exploration Algorithm -
    # The exploration algorithm helps the RL Agent to explore the environment by occasionally choosing suboptimal
    # actions or giving reward bonuses to unseen states instead of exploiting the current knowledge.
    # Choose from "None", "EpsilonGreedy", "ICM" and "RND"
    exploration_algorithm = args.exploration_algorithm

    # - Preprocessing Algorithm -
    # In some cases it can be useful to present not just the raw data from the environment to the RL Agent but to
    # apply some preprocessing first. In case of Semantic Segmentation, a previously trained Variational Autoencoder
    # takes the image input and transforms it into a much more compact representation leading to faster convergence.
    # Choose from "None" and "SemanticSegmentation"
    preprocessing_algorithm = args.preprocessing_algorithm
    # Enter the path for the preprocessing model if needed
    preprocessing_path = args.preprocessing_path

    # - Misc -
    # Determine if old model checkpoints will be overwritten or kept.
    trainer.remove_old_checkpoints = args.remove_old_checkpoints

    # - Self-Play Tournament -
    trainer.games_per_fixture = args.games_per_fixture
    trainer.game_results_path = args.game_results_path
    trainer.rating_history_path = args.rating_history_path
    
    # endregion

    # region --- Trainer Initialization ---

    # Given a path with one or more trained models the following function creates a dictionary that contains a unique
    # key for each model along with the corresponding model paths, the number of steps it's been trained for, the reward
    # it reached and possibly a (elo) rating. The same is done for a clone of the original model in case of self-play.
    trainer.create_model_dictionaries(model_path, clone_path)
    # With the model dictionaries create a tournament schedule can be created where each model plays against every
    # other model. This is only necessary for self-play environments in tournament mode.
    trainer.create_tournament_schedule()

    # Parse the trainer configuration (make sure to select the right key)
    trainer.parse_training_parameters(args.training_parameters[0], args.training_parameters[1])
    # Instantiate the agent which consists of a learner and one or multiple actors
    print()
    trainer.async_instantiate_agent(mode, preprocessing_algorithm, exploration_algorithm,
                                    environment_path, preprocessing_path, demonstration_path, vars(args))
    # If you are trying to understand this project, the next place to continue exploring it would be the trainer file
    # in the respective directory (./modules/trainer.py)

    # endregion

    # region --- Training / Testing ---
    # In training mode the actors play new episodes while the learner trains neural networks based on experiences
    # until the program is cancelled manually.
    if mode == "training":
        trainer.async_training_loop()
    # In testing mode the actors play with already trained models until the program is cancelled manually. There is no
    # improvement since the learner is not doing anything.
    elif mode == "testing" or mode == "fastTesting":
        trainer.async_testing_loop()
    # The tournament loop plays all matches in the respective schedule and writes their results into a csv file.
    else:
        trainer.async_tournament_loop()
    # endregion


if __name__ == '__main__':
    main()
