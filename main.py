# ----------------------------------------------------------------------------------
# >>> Modular Reinforcement Learning© created and maintained by Pascal Graf 2021 <<<
# ----------------------------------------------------------------------------------

# region --- Imports ---
import numpy as np
from modules.trainer.trainer import Trainer
np.set_printoptions(precision=2)
# endregion


def main():
    """ Modular Reinforcement Learning© main-function
    This function defines the parameters to instantiate a new trainer object which then creates one or multiple actors
    and learners to connect to an Unity or OpenAI Gym environment. After that the main-function starts the training
    or testing procedure.

    Before running this function, please adjust the parameters in the Parameter Choice region as well as the trainer
    configuration under the respective key (./trainer_configs/trainer_config.yaml).

    :return: None
    """

    # region  --- Parameter Choice ---

    # Choose between "training", "testing" or "fastTesting"
    # If you want to test a trained model or continue learning from a checkpoint enter the model path below
    mode = "testing"
    model_path = r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\211205_103919_SAC_AirHockey_ShotOnGoal_New" # r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\211203_070629_SAC_AirHockey_ShotOnGoal" #r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\211201_101350_SAC_Worm_PER0.8_Recurrent60_10Actors_0.3Eps" # r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\211122_150626_SAC_Crawler"#r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\211025_173951_SAC_Worm_PER_BurnIn_Recurrent"

    # Instantiate a Trainer object with certain choices of parameters and algorithms
    trainer = Trainer()
    interface = 'MLAgentsV18'  # Choose from "MLAgentsV18" (Unity) and "OpenAIGym"
    # If you want to run multiple Unity actors in parallel you need specify the path to the '.exe' file here.
    environment_path = r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\AirHockeyFullGame\AirHockeyRebuild.exe" #r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\SingleWorm\UnityEnvironment.exe" #r"C:\PGraf\Arbeit\RL\EnvironmentBuilds\RobotArm\Grabbing\Level0\DoBotEnvironment.exe" # In case of "OpenAIGym" enter the desired env name here, e.g. "LunarLanderContinuous-v2"

    # Choose from "None", "EpsilonGreedy", "ICM" and "RND"
    exploration_algorithm = "EpsilonGreedy"

    # Choose from "DQN", "DDPG", "TD3", "SAC"
    trainer.select_training_algorithm('SAC')

    # Choose from None, "LinearCurriculum" ("RememberingCurriculum" and "CrossFadeCurriculum" are currently disabled)
    curriculum_strategy = None
    trainer.select_curriculum_strategy(curriculum_strategy)

    # Choose from "None" and "SemanticSegmentation"
    preprocessing_algorithm = 'None'
     # Enter the path for the preprocessing model if needed
    preprocessing_path = r"C:\PGraf\Arbeit\RL\SemanticSegmentation\vae\models\210809_101443_VAE_encoder_235"

    trainer.save_all_models = True  # Determines if all models or only the actor will be saved during training
    trainer.remove_old_checkpoints = False  # Determines if old model checkpoints will be overwritten

    # endregion

    # region --- Initialization ---

    # Parse the trainer configuration (make sure to select the right key)
    trainer.parse_training_parameters("trainer_configs/trainer_config.yaml", "sac")
    # Instantiate the agent which consists of a learner and one or multiple actors
    trainer.async_instantiate_agent(mode, interface, preprocessing_algorithm, exploration_algorithm,
                                    environment_path, model_path, preprocessing_path)

    # endregion

    # region --- Training / Testing ---

    # Play new episodes until the training/testing is manually interrupted
    if mode == "training":
        trainer.async_training_loop()
    else:
        trainer.async_testing_loop()
    # endregion


if __name__ == '__main__':
    main()
