# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2023 <<<
# ----------------------------------------------------------------------------------

# region --- Imports ---
import numpy as np
from modules.trainer import Trainer
from modules.misc.cmd_arg_parse import CmdArgParse
from modules.training_algorithms.minimal_agent_blueprint import MinimalActor as Actor
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
    # If you want to test a trained model or continue learning from a checkpoint enter the model path via command line
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

    # - Preprocessing Algorithm -
    # In some cases it can be useful to present not just the raw data from the environment to the RL Agent but to
    # apply some preprocessing first. In case of Semantic Segmentation, a previously trained Variational Autoencoder
    # takes the image input and transforms it into a much more compact representation leading to faster convergence.
    # Choose from "None" and "SemanticSegmentation"
    preprocessing_algorithm = args.preprocessing_algorithm
    # Enter the path for the preprocessing model if needed
    preprocessing_path = args.preprocessing_path
    # endregion

    # region --- Trainer Initialization ---

    # Given a path with one or more trained models the following function creates a dictionary that contains a unique
    # key for each model along with the corresponding model paths, the number of steps it's been trained for, the reward
    # it reached and possibly a (elo) rating. The same is done for a clone of the original model in case of self-play.
    trainer.create_model_dictionaries(model_path, clone_path)

    # Parse the trainer configuration (make sure to select the right key)
    trainer.parse_training_parameters(args.training_parameters[0], args.training_parameters[1])
    # Instantiate the agent which consists of an actor
    actor_num = 1
    # endregion

    # region - Exploration Parameter Determination and Network Feedback -
    # The exploration is a list that contains a dictionary for each actor (if multiple).
    # Each dictionary contains a value for beta and gamma (utilized for intrinsic motivation)
    # as well as a scaling value (utilized for epsilon greedy). If there is only one actor in training mode
    # the scaling of exploration should start at maximum.
    exploration_degree = [{'beta': 0, 'gamma': 0.99, 'scaling': 0}]

    # Pass the exploration degree to the trainer configuration
    trainer_configuration["ExplorationParameters"]["ExplorationDegree"] = exploration_degree
    # endregion

    # region - Actor Instantiation and Environment Connection -
    # Create the desired number of actors using the ray "remote"-function. Each of them will construct their own
    # environment, exploration algorithm and preprocessing algorithm instances. The actors are distributed along
    # the cpu cores and threads, which means their number in the trainer configuration should not extend the maximum
    # number of cpu threads - 2.
    actor = Actor(idx, 5004 + idx, mode,
                  self.interface,
                  preprocessing_algorithm,
                  preprocessing_path,
                  exploration_algorithm,
                  environment_path,
                  demonstration_path,
                  '/cpu:0')

    # Instantiate one environment for each actor and connect them to one another.
    if interface == "MLAgentsV18":
        actor.connect_to_unity_environment()
    else:
        actor.connect_to_gym_environment()
    actor.instantiate_modules(trainer_configuration)
    # In case of Unity Environments set the rendering and simulation parameters.
    if interface == "MLAgentsV18":
        actor.set_unity_parameters(time_scale=1, width=500, height=500)
    # endregion

    # region - Environment & Exploration Configuration Query -
    # Get the environment and exploration configuration from the first actor.
    environment_configuration = actor.get_environment_configuration()
    # endregion

    # region - Learner Instantiation and Network Construction -
    # Initialize the actor network for each actor
    actor.load_network()
    # endregion
    # endregion
    # endregion

    # region --- Testing ---
    # In testing mode the actors play with already trained models until the program is cancelled manually. There is no
    # improvement since the learner is not doing anything.
    testing_loop()
    # endregion


if __name__ == '__main__':
    main()
