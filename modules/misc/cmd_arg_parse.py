import argparse

class CmdArgParse:
    def return_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode', help="Choose between 'training', 'testing' or 'fastTesting'", type=str,
                            required=False, default="training")
        parser.add_argument('-mp', '--model_path', help="If you want to test a trained model "
                                                        "or continue learning from a "
                                                        "checkpoint enter the model path here.",
                            type=str, required=False, default=None)
        parser.add_argument('-cp', '--clone_path', help="If defined this path defines the clone's weights for selfplay "
                                                        "training / testing. Otherwise, model_path will be used.",
                            type=str, required=False, default=None)
        parser.add_argument('-ti', '--trainer_interface', help="Choose from 'MLAgentsV18' (Unity) and 'OpenAIGym'",
                            type=str, required=False, default='MLAgentsV18')
        parser.add_argument('-ep', '--env_path', help="If you want to run multiple Unity actors in parallel you need to"
                                                      " specify the path to the Environment '.exe' here. In case of "
                                                      "'OpenAIGym 'enter the desired env name here instead, "
                                                      "e.g.'LunarLanderContinuous-v2'. If you want a CQL agent to "
                                                      "learn from demonstrations, an environment can be used to "
                                                      "evaluate the model on a regular basis. Please provide a path "
                                                      "or typeNone to connect directly to the Unity Editor. "
                                                      "Otherwise, type 'NoEnv' to proceed without evaluation.",
                            type=str, required=False, default=None)
        parser.add_argument('-ta', '--train_algorithm', help="Choose from 'DQN', 'DDPG', 'TD3', 'SAC', 'CQL'", type=str,
                            required=False, default='SAC')
        parser.add_argument('-dp', '--demo_path', help="In case you want to train the agent offline via "
                                                       "CQL please provide the path for demonstrations.",
                            type=str, required=False, default=None)
        parser.add_argument('-ea', '--exploration_algorithm', help="Choose from 'None', 'EpsilonGreedy', 'ICM', 'RND', "
                                                                   ", 'RNDAlter', 'ENM', 'NGU', 'ECR', 'NGUr'",
                            type=str, required=False, default='NGU')
        parser.add_argument('-me', '--meta_learning_algorithm', help="Choose from 'None', 'MetaController'",
                            type=str, required=False, default='MetaController')
        parser.add_argument('-cs', '--curriculum_strategy', help="Choose from None, 'LinearCurriculum', "
                                                                 "('RememberingCurriculum', 'CrossFadeCurriculum' "
                                                                 "are currently disabled)",
                            type=str, required=False, default=None)
        parser.add_argument('-pa', '--preprocessing_algorithm', help="Choose from 'None' and 'SemanticSegmentation'",
                            type=str, required=False, default='None')
        parser.add_argument('-pp', '--preprocessing_path', help="Enter the path for the preprocessing model if needed",
                            type=str,required=False, default='None')
        parser.add_argument('-sm', '--save_all_models', help="Determines if all models or only the actor will be saved "
                                                             "during training", type=bool, required=False, default=True)
        parser.add_argument('-rmoc', '--remove_old_checkpoints', help="Determines if old model checkpoints will "
                                                                      "be overwritten",
                            type=bool, required=False, default=False)
        parser.add_argument('-p', '--training_parameters', help="Parse the trainer configuration (make sure to select "
                                                                "the right key (training algorithm) "
                                                                "e.g. <path to yaml> <key>)", type=str, nargs=2,
                            required=False, default=['trainer_configs/trainer_config.yaml', 'sac'])

        args = parser.parse_args()
        return args