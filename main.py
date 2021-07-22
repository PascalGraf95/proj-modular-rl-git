from modules.trainer.trainer import Trainer
import numpy as np
import streamlit as st
import pandas as pd
import tensorflow as tf
st.set_option('deprecation.showfileUploaderEncoding', False)
np.set_printoptions(precision=2)


def main():
    # region  --- Parameter Choice ---
    # 1. Choose between Training and Testing
    mode = "testing"
    model_path = r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\210715_072854_SAC_RobotGrabbingCrossFade"#r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\210713_183619_SAC_RobotGrabbingCrossFade"#r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-modular-reinforcement-learning\training\summaries\210712_144215_SAC_RobotGrabbingRememberingCurriculum"
    time_scale = 1000
    # 2. Instantiate Trainer
    trainer = instantiate_trainer()
    # trainer.logging_frequency = 100
    # 3. Choose Interface, Exploration and Training Algorithm
    # trainer.change_interface('OpenAIGym')
    # trainer.environment_selection = "LunarLanderContinuous-v2"
    trainer.change_interface('MLAgentsV18')
    trainer.change_exploration_algorithm('None')
    trainer.change_training_algorithm('SAC')
    trainer.change_curriculum_strategy('CrossFadeCurriculum')
    trainer.change_preprocessing_algorithm('None')

    # 4. Get and Validate Configurations
    trainer.get_agent_configuration()
    trainer.get_exploration_configuration()
    trainer.parse_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml", "sac")
    print(trainer.validate_trainer_configuration())

    # endregion

    # region --- Initialization ---
    # Instantiate Curriculum Strategy
    trainer.instantiate_curriculum_strategy()

    # 5. Connect to the Environment and set/get its Configuration
    trainer.connect()
    trainer.get_environment_configuration()
    trainer.instantiate_preprocessing_algorithm(r"C:\PGraf\Arbeit\RL\MarkerDetection")

    # Set Unity Parameters
    if mode == "training" or mode == "fastTesting":
        trainer.set_unity_parameters(time_scale=time_scale)
    else:
        trainer.set_unity_parameters(time_scale=1.0)

    # 6. Instantiation
    if mode == "training":
        trainer.instantiate_agent('training', model_path)
        trainer.instantiate_logger(tensorboard=True)
    else:
        trainer.instantiate_agent('testing', model_path)
        trainer.instantiate_logger(tensorboard=False)
    trainer.save_all_models = True
    trainer.remove_old_checkpoints = False
    trainer.instantiate_exploration_algorithm()
    trainer.instantiate_replay_buffer()
    # endregion

    # region --- Training/Testing Algorithm ---
    while True:
        if mode == "training":
            mean_episode_length, mean_episode_reward, \
                episodes, training_step, loss, \
                number_of_episodes_for_average, task_level, reward_average = next(trainer.training_loop)
        else:
            mean_episode_length, mean_episode_reward, \
                episodes, training_step, loss, \
                number_of_episodes_for_average, task_level, reward_average = next(trainer.testing_loop)
        training_duration = trainer.get_elapsed_training_time()

        if episodes - trainer.last_debug_message >= 10:
            trainer.last_debug_message = episodes
            print("Played Episodes: {}, Training Steps: {}, Task Level: {:d}\n"
                  "Average Episode Reward: {:.2f} (for the last {:d} Episodes)\n"
                  "Elapsed Training Time: {:02d}:{:02d}:{:02d}:{:02d}".format(episodes,
                                                                              training_step,
                                                                              int(task_level),
                                                                              reward_average,
                                                                              int(number_of_episodes_for_average),
                                                                              *training_duration))
            print("-------------------------------------------------")
    # endregion


@st.cache(allow_output_mutation=True)
def instantiate_trainer():
    return Trainer()


def print_parameter_mismatches(mis_par, obs_par, mis_net_par, obs_net_par, mis_expl_par, obs_expl_par,
                               wro_type, wro_type_net, wro_type_expl):
    successful = True
    if len(mis_par):
        successful = False
        st.error("The following parameters which are requested by the agent "
                 "are not defined in the trainer configuration: {}".format(", ".join(mis_par)))
    if len(obs_par):
        st.warning("The following parameters are not requested by the agent "
                   "but are defined in the trainer configuration: {}".format(", ".join(obs_par)))

    if len(mis_net_par):
        successful = False
        st.error("The following network parameters which are requested by the agent "
                 "are not defined in the trainer configuration: {}".format(", ".join(mis_net_par)))
    if len(obs_net_par):
        st.warning("The following network parameters are not requested by the agent "
                   "but are defined in the trainer configuration: {}".format(", ".join(obs_net_par)))

    if len(mis_expl_par):
        successful = False
        st.error("The following exploration parameters which are requested by the algorithm "
                 "are not defined in the trainer configuration: {}".format(", ".join(mis_expl_par)))
    if len(obs_expl_par):
        st.warning("The following exploration parameters are not requested by the algorithm "
                   "but are defined in the trainer configuration: {}".format(", ".join(obs_expl_par)))

    if len(wro_type):
        successful = False
        st.error("The following training parameters do not match the "
                 "data types requested by the agent: {}".format(", ".join(wro_type)))

    if len(wro_type_net):
        successful = False
        st.error("The following network parameters do not match the data types requested by "
                 "the agent network: {}".format(", ".join(wro_type_net)))

    if len(wro_type_net):
        successful = False
        st.error("The following exploration parameters do not match the data types requested by "
                 "the algorithm: {}".format(", ".join(wro_type_expl)))
    return successful


def streamlit_app():
    # region --- Title and Trainer Instantiation ---
    st.set_page_config(page_title="Modular Reinforcement Learning",
                            page_icon=":ant:", layout='centered', initial_sidebar_state='auto')
    st.write('''# Modular Reinforcement Learning''')
    st.write('''<sub>Created and maintained by [Pascal Graf](mailto:pascal.graf@hs-heilbronn.de) (2021)</sub>''',
             unsafe_allow_html=True)
    # Create Trainer
    trainer = instantiate_trainer()
    # endregion

    # region --- Project Information and Reset Options ---
    placeholder_project_information = st.empty()
    with placeholder_project_information.beta_container():
        # if st.checkbox('Get more information about this project.'):
        with st.beta_expander('Get more information about this project.'):
            st.info("This Modular Reinforcement Learning tool was created to allow for fast and easy prototyping "
                    "as well as comparison of different RL algorithms and environments. Furthermore it features "
                    "multiple exploration algorithms. New Reinforcement Learning methods and exploration strategies "
                    "can be implemented by utilizing and inheriting from the respective blueprint classes.")
        col_reset_trainer, col_update_page = st.beta_columns(2)
        with col_reset_trainer:
            if st.button("Reset Trainer"):
                trainer.reset()
        with col_update_page:
            st.button("Update Page")
        st.write('''<hr>''', unsafe_allow_html=True)
    # endregion

    # region --- Algorithm Choice & Instantiation ---
    placeholder_algorithm_choice = st.empty()
    if trainer.initiation_progress == 0:
        with placeholder_algorithm_choice.beta_container():
            st.write('''## Training Component Selection:''')
            with st.beta_expander('Get more information about the component selection.'):
                st.info('''
            Here you can choose between different implemented RL algorithms, curriculum strategies and exploration 
            techniques. Additionally, you have to select a trainer configuration 
            (which can be altered in the next initialization step) and an interface depending on which type of 
            environment you plan to work with.
                ''')
            trainer_config_keys = trainer.parse_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml")
            col_trainer_config_keys, col_trainer_config_delete = st.beta_columns(2)
            with col_trainer_config_keys:
                trainer.config_key = st.selectbox(label="Select a Trainer Configuration:", options=trainer_config_keys)
            with col_trainer_config_delete:
                if st.button("Delete Trainer Configuration"):
                    trainer.delete_training_configuration("modules/trainer/trainer_configs/trainer_config.yaml",
                                                          trainer.config_key)
                    trainer.reset()
            col_interface_name, col_exploration_name, col_rl_name = st.beta_columns(3)

            with col_interface_name:
                interface_name = st.selectbox(label="Select an Interface:", options=["MLAgentsV18", "OpenAIGym"])
            with col_exploration_name:
                exploration_name = st.selectbox(label="Select an Exploration Algorithm:", options=["None",
                                                                                                   "EpsilonGreedy"])
            with col_rl_name:
                rl_name = st.selectbox(label="Select a RL Algorithm:", options=["DQN", "A2C", "DDPG", "TD3", "SAC"])
            if interface_name == "OpenAIGym":
                trainer.environment_selection = st.selectbox(
                    label="Select an OpenAI Gym Environment:",
                    options=["Acrobot-v1", "CartPole-v1", "MountainCar-v0",
                             "MountainCarContinuous-v0", "Pendulum-v0", "BipedalWalker-v3",
                             "BipedalWalkerHardcore-v3", "CarRacing-v0", "LunarLander-v2",
                             "LunarLanderContinuous-v2", "FetchPickAndPlace-v1"])

            curriculum_strategy = st.selectbox(label="Select an Curriculum Learning Strategy",
                                               options=["None",
                                                        "LinearCurriculum",
                                                        "RememberingCurriculum",
                                                        "CrossFadeCurriculum"])
            if st.button("Confirm Selections"):
                # Change to selected components
                trainer.change_interface(interface_name)
                trainer.change_exploration_algorithm(exploration_name)
                trainer.change_training_algorithm(rl_name)
                trainer.change_curriculum_strategy(curriculum_strategy)

                # Get configurations
                trainer.get_agent_configuration()
                trainer.get_exploration_configuration()
                trainer.parse_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml",
                                                  trainer.config_key)
                trainer.initiation_progress = 1
    # endregion

    # region --- Training Parameter Modification ---
    placeholder_training_parameters = st.empty()
    if trainer.initiation_progress == 1:
        with placeholder_training_parameters.beta_container():
            placeholder_algorithm_choice.empty()
            st.success("Successfully instantiated training components.")
            st.write('''## Training Parameter Modification:''')
            with st.beta_expander('Get more information about parameter modifications.'):
                st.info("In this part of the UI you can modify the training and network parameters for the chosen "
                        "Reinforcement Learning and Exploration algorithms. The changes will be saved under the "
                        "respective configuration key in the Trainer Configuration .yaml-File. In order to proceed to"
                        "the training you need to add all the missing parameters listed in the error message below "
                        "(if existent). Warning messages don't need to be bothered about.")
            with st.beta_expander('Modify Training Parameters'):
                config_key = st.text_input("Trainer Configuration Key (Param)", value=trainer.config_key)
                idx = 0
                for key, val in trainer.agent_configuration['TrainingParameterSpace'].items():
                    if not idx % 2:
                        cols = st.beta_columns(2)
                    with cols[idx % 2]:
                        if val == str:
                            trainer.trainer_configuration[key] = \
                                st.text_input(key, value=trainer.trainer_configuration.get(key, ""))
                        elif val == int:
                            trainer.trainer_configuration[key] = \
                                st.number_input(key, value=trainer.trainer_configuration.get(key, 0))
                        elif val == float:
                            trainer.trainer_configuration[key] = \
                                st.number_input(key, value=trainer.trainer_configuration.get(key, 0.0),
                                                step=0.00001, min_value=-5.0, max_value=1000.0, format="%f")
                        elif val == bool:
                            trainer.trainer_configuration[key] = \
                                st.checkbox(key,
                                            value=trainer.trainer_configuration.get(key, False))
                        else:
                            idx -= 1
                        idx += 1

                if st.button("Save Parameter Changes"):
                    if trainer.save_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml",
                                                        config_key):
                        trainer.config_key = config_key
                        st.success("Successfully edited the Trainer Configuration.")
            with st.beta_expander('Modify Network Parameters'):
                config_key = st.text_input("Trainer Configuration Key (Net)", value=trainer.config_key)
                for idx, network_dict in enumerate(trainer.agent_configuration['NetworkParameterSpace']):
                    st.write('''### {}'''.format(trainer.agent_configuration["NetworkTypes"][idx]))
                    index = 0
                    for key, val in network_dict.items():
                        if not index % 2:
                            cols = st.beta_columns(2)
                        with cols[index % 2]:
                            if val == str:
                                trainer.trainer_configuration["NetworkParameters"][idx][key] = \
                                    st.text_input(key + " " + str(idx+1),
                                                  value=trainer.trainer_configuration["NetworkParameters"][idx].get(key, ""))
                            elif val == int:
                                trainer.trainer_configuration["NetworkParameters"][idx][key] = \
                                    st.number_input(key + " " + str(idx+1),
                                                    value=trainer.trainer_configuration["NetworkParameters"][idx].get(key, 0))
                            elif val == float:
                                trainer.trainer_configuration["NetworkParameters"][idx][key] = \
                                    st.number_input(key + " " + str(idx+1),
                                                    value=trainer.trainer_configuration["NetworkParameters"][idx].get(key, 0.0),
                                                    step=0.00001, min_value=0.0, max_value=1000.0, format="%f")
                            elif val == bool:
                                trainer.trainer_configuration["NetworkParameters"][idx][key] = \
                                    st.checkbox(key + " " + str(idx+1),
                                                value=trainer.trainer_configuration["NetworkParameters"][idx].get(key, False))
                            else:
                                index -= 1
                        index += 1
                if st.button("Save Network Changes"):
                    if trainer.save_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml", config_key):
                        trainer.config_key = config_key
                        st.success("Successfully edited the Trainer Configuration.")
            with st.beta_expander('Modify Exploration Parameters'):
                config_key = st.text_input("Trainer Configuration Key (Exploration)", value=trainer.config_key)
                idx = 0
                for key, val in trainer.exploration_configuration["ParameterSpace"].items():
                    if not idx % 2:
                        cols = st.beta_columns(2)
                    with cols[idx % 2]:
                        if val == str:
                            trainer.trainer_configuration["ExplorationParameters"][key] = \
                                st.text_input(key,
                                              value=trainer.trainer_configuration["ExplorationParameters"].get(key, ""))
                        elif val == int:
                            trainer.trainer_configuration["ExplorationParameters"][key] = \
                                st.number_input(key,
                                                value=trainer.trainer_configuration["ExplorationParameters"].get(key, 0))
                        elif val == float:
                            trainer.trainer_configuration["ExplorationParameters"][key] = \
                                st.number_input(key,
                                                value=trainer.trainer_configuration["ExplorationParameters"].get(key, 0.0),
                                                step=0.00001, min_value=0.0, max_value=1000.0, format="%f")
                        elif val == bool:
                            trainer.trainer_configuration["ExplorationParameters"][key] = \
                                st.checkbox(key, value=trainer.trainer_configuration["ExplorationParameters"].get(key, False))
                        else:
                            idx -= 1
                        idx += 1
                if st.button("Save Exploration Changes"):
                    if trainer.save_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml",
                                                        config_key):
                        trainer.config_key = config_key
                        st.success("Successfully edited the Trainer Configuration.")
            print_parameter_mismatches(*trainer.validate_trainer_configuration())
            if st.button("Confirm Parameters"):
                trainer.parse_training_parameters("modules/trainer/trainer_configs/trainer_config.yaml", trainer.config_key)
                if print_parameter_mismatches(*trainer.validate_trainer_configuration()):
                    trainer.initiation_progress = 2
            st.write('''<hr>''', unsafe_allow_html=True)
    # endregion

    # region --- Environment Connection ---
    placeholder_environment_connection = st.empty()
    if trainer.initiation_progress == 2:
        with placeholder_environment_connection.beta_container():
            placeholder_training_parameters.empty()
            st.success("Successfully set training parameters.")
            st.write('''## Environment Connection:''')
            with st.beta_expander('Get more information about the environment connection.'):
                st.info("Here you can connect the trainer to the desired environment. Depending on your interface "
                        "choice in the Algorithm Instantiation section this will either open an OpenAI Gym environment"
                        "or connect to the Unity Editor (press Play in Unity after pressing the Connect button).")
            if st.button("Connect to Environment"):
                trainer.instantiate_curriculum_strategy()
                trainer.connect()
                trainer.get_environment_configuration()
                if trainer.validate_action_space():
                    trainer.initiation_progress = 3
                else:
                    st.error("The action spaces of the environment and the agent are not compatible.")
    # endregion

    # region --- Training / Testing ---
    # region --- Environment Control ---
    placeholder_environment_control = st.empty()
    if trainer.initiation_progress >= 3:
        with placeholder_environment_control.beta_container():
            placeholder_environment_connection.empty()
            st.write('''## Environment Control:''')
            if st.button("Close Environment"):
                trainer.env.close()
                trainer.initiation_progress = 2
    # endregion
    # region --- Choose between Training and Testing ---
    placeholder_training_testing = st.empty()
    if trainer.initiation_progress == 3:
        with placeholder_training_testing.beta_container():
            st.success("Successfully connected to Unity Environment.")
            st.write('''## Process Selection:''')
            col_start_training, col_continue_training, col_start_testing = st.beta_columns(3)
            model_path = st.text_input("Model Weights Path:", value=trainer.model_path)
            with col_start_training:
                if st.button("Start New Training"):
                    if not trainer.environment_selection:
                        # Unity Parameters
                        trainer.set_unity_parameters(time_scale=1000.0)

                    # Instantiation
                    trainer.instantiate_agent('training', '')
                    trainer.instantiate_logger()
                    trainer.instantiate_exploration_algorithm()
                    trainer.instantiate_replay_buffer()

                    # Step
                    trainer.initiation_progress = 4
            with col_continue_training:
                if st.button("Continue Training"):
                    trainer.model_path = model_path
                    if not trainer.environment_selection:
                        # Unity Parameters
                        trainer.set_unity_parameters(time_scale=1000.0)

                    # Instantiation
                    trainer.instantiate_agent('training', trainer.model_path)
                    trainer.instantiate_logger()
                    trainer.instantiate_exploration_algorithm()
                    trainer.instantiate_replay_buffer()

                    # Step
                    trainer.initiation_progress = 4
            with col_start_testing:
                if st.button("Start Testing"):
                    trainer.model_path = model_path
                    if not trainer.environment_selection:
                        trainer.set_unity_parameters(time_scale=1.0)
                    # Instantiation
                    trainer.instantiate_agent('testing', trainer.model_path)
                    trainer.instantiate_logger(tensorboard=False)
                    trainer.instantiate_exploration_algorithm()
                    trainer.instantiate_replay_buffer()

                    # Step
                    trainer.initiation_progress = 5
    # endregion
    # region --- Training in Progress ---
    if trainer.initiation_progress == 4:
        training_progress_text = st.empty()
        col_break_button, col_boost_exploration_button, col_save_all, col_remove_old = st.beta_columns(4)
        with col_break_button:
            break_training_loop = st.button("Stop Training Process")
        with col_boost_exploration_button:
            boost_exploration = st.button("Boost Exploration")
        with col_save_all:
            if st.checkbox("Save all Models", False):
                trainer.save_all_models = True
            else:
                trainer.save_all_models = False
        with col_remove_old:
            if st.checkbox("Remove old Model Checkpoints", False):
                trainer.remove_old_checkpoints = True
            else:
                trainer.remove_old_checkpoints = False

        # Training loop
        while True:
            # Check for training interruption through button press
            if break_training_loop:
                st.warning("Training was manually interrupted!")
                trainer.env.close()
                trainer.initiation_progress = 2
                break
            if boost_exploration:
                trainer.boost_exploration()
                boost_exploration = False
            try:
                # Gather next yield from training loop generator
                mean_episode_length, mean_episode_reward, \
                episodes, training_step, loss, \
                number_of_episodes_for_average, task_level, reward_average = next(trainer.training_loop)

                training_duration = trainer.get_elapsed_training_time()

                # Print info text for last training episode
                training_progress_text.info(
                    '''
                        Played Episodes: {}, Training Steps: {}, Task Level: {}\n
                        Average Episode Reward: {:.2f} (for the last {:d} Episodes)\n
                        Elapsed Training Time: {:02d}:{:02d}:{:02d}:{:02d}\n                        
                    '''.format(episodes,
                               training_step,
                               task_level,
                               reward_average,
                               number_of_episodes_for_average,
                               *training_duration))

            # Throw no error when training ended due to reached number of episodes or early stopping
            except StopIteration:
                print("Training completed!")
                break
    # endregion
    # region --- Testing in Progress ---
    if trainer.initiation_progress == 5:
        # Clear previous UI section
        placeholder_training_testing.empty()

        testing_progress_text = st.empty()
        break_testing_loop = st.button("Stop Testing")
        # Testing Loop
        while True:
            if break_testing_loop:
                st.warning("Testing was manually interrupted!")
                trainer.env.close()
                trainer.initiation_progress = 2
                break
                # Gather next yield from training loop generator
            mean_episode_length, mean_episode_reward, \
            episodes, training_step, loss, \
            number_of_episodes_for_average, task_level, reward_average = next(trainer.training_loop)

            training_duration = trainer.get_elapsed_training_time()

            # Print info text for last training episode
            testing_progress_text.info(
                '''
                    Played Episodes: {}, Training Steps: {}, Task Level: {}\n
                    Average Episode Reward: {:.2f} (for the last {:d} Episodes)\n
                    Elapsed Testing Time: {:02d}:{:02d}:{:02d}:{:02d}\n                        
                '''.format(episodes,
                           training_step,
                           task_level,
                           reward_average,
                           number_of_episodes_for_average,
                           *training_duration))
    # endregion
    # endregion


if __name__ == '__main__':
    main()
    # streamlit_app()
