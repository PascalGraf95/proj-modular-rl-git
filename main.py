# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2022 <<<
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

    parser = CmdArgParse()
    args = parser.return_args()
    # region  --- Parameter Choice ---

    # Choose between "training", "testing" or "fastTesting"
    # If you want to test a trained model or continue learning from a checkpoint enter the model path below
    mode = args.mode
    model_path = args.model_path #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220913_161201_SAC_Airhockey_Selfplay_TrainingBasicBehavior" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220903_124426_SAC_Airhockey_Selfplay_Multiagent" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220901_150641_SAC_Airhockey_Selfplay_Multiagent" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220830_145608_SAC_Airhockey_Selfplay_Multiagent" # r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220823_165533_SAC_Airhockey_Selfplay_Multiagent" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220712_131840_SAC1" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220712_131840_SAC2" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220708_160314_SAC_Airhockey_Selfplay" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220708_143215_SAC_Airhockey_Selfplay" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220527_094925_CQL_SinglePendulum" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220525_092958_CQL_SinglePendulum" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220524_133838_CQL_SinglePendulum" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220514_131626_SAC_InvertedDoublePendulumCart_Recurrent"
    # If defined this path defines the clone's weights for selfplay training/testing. Otherwise, model_path will be
    # used.
    clone_path = args.clone_path #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220903_124426_SAC_Airhockey_Selfplay_Multiagent" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\220712_131840_SAC1"

    # Instantiate a Trainer object with certain choices of parameters and algorithms
    trainer = Trainer()
    trainer.interface = args.trainer_interface  # Choose from "MLAgentsV18" (Unity) and "OpenAIGym"
    # If you want to run multiple Unity actors in parallel you need to specify the path to the Environment '.exe' here.
    # In case of "OpenAIGym" enter the desired env name here instead, e.g. "LunarLanderContinuous-v2".
    # If you want a CQL agent to learn from demonstrations, an environment can be used to evaluate the model on a
    # regular basis. Please provide a path or type None to connect directly to the Unity Editor. Otherwise, type
    # 'NoEnv' to proceed without evaluation.
    environment_path = args.env_path #r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\3DAirHockeyMujoco" #r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\3DAirHockeyMujoco" #r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\3DAirHockeyMujoco" #r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\RobotArm\Conveyor\DoBotEnvironment.exe"

    # - Training Algorithm -
    # This is the core learning algorithm behind the RL Agent. While Deep Q-Learning / Deep Q Networks (DQN) presumably
    # is the most famous algorithm it can only act in environments with discrete action space. The three others
    # in their current implementation only support continuous action spaces. Of those three Soft Actor-Critic (SAC)
    # is the most recent and preferred option.
    # Choose from "DQN", "DDPG", "TD3", "SAC", "CQL"
    trainer.select_training_algorithm(args.train_algorithm)
    # In case you want to train the agent offline via CQL please provide the path for demonstrations.
    demonstration_path = args.demo_path 

    # - Exploration Algorithm -
    # The exploration algorithm helps the RL Agent to explore the environment by occasionally choosing suboptimal
    # actions or giving reward bonuses to unseen states instead of exploiting the current knowledge.
    # Choose from "None", "EpsilonGreedy", "ICM" and "RND"
    exploration_algorithm = "None"

    # - Curriculum Strategy -
    # Just like humans, a RL Agent learns best by steadily increasing the difficulty of the given task. Thus, for
    # Unity Environments an option to choose a task level has been implemented. Notice, however, that in order to work
    # the Unity environment needs special preparation (like a SideChannel reading the messages).
    # Choose from None, "LinearCurriculum" ("RememberingCurriculum" and "CrossFadeCurriculum" are currently disabled)
    curriculum_strategy = args.curriculum_strategy
    trainer.select_curriculum_strategy(curriculum_strategy)

    # - Preprocessing Algorithm -
    # In some cases it can be useful to present not just the raw data from the environment to the RL Agent but to
    # apply some preprocessing first. In case of Semantic Segmentation, a previously trained Variational Autoencoder
    # takes the image input and transforms it into a much more compact representation leading to faster convergence.
    # Choose from "None" and "SemanticSegmentation"
    preprocessing_algorithm = args.preprocessing_algorithm
    # Enter the path for the preprocessing model if needed
    preprocessing_path = args.preprocessing_path

    # - Misc -
    trainer.save_all_models = True  # Determines if all models or only the actor will be saved during training
    trainer.remove_old_checkpoints = False  # Determines if old model checkpoints will be overwritten

    # endregion

    # region --- Initialization ---

    # Parse the trainer configuration (make sure to select the right key)
    trainer.parse_training_parameters(args.training_parameters[0], args.training_parameters[1])
    # Instantiate the agent which consists of a learner and one or multiple actors
    trainer.async_instantiate_agent(mode, preprocessing_algorithm, exploration_algorithm,
                                    environment_path, model_path, preprocessing_path, demonstration_path, clone_path)
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
