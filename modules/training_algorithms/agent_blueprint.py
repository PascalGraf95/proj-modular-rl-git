#!/usr/bin/env python
import numpy as np


class Agent:
    # Static, algorithm specific Parameters
    TrainingParameterSpace = {
        'TrainingID': str,
        'Episodes': int,
        'BatchSize': int,
        'Gamma': float,
        'TrainingInterval': int,
        'NetworkParameters': list,
        'ExplorationParameters': dict,
        'TrainingDescription': str
    }
    NetworkParameterSpace = []
    ActionType = []
    ReplayBuffer = False
    LearningBehavior = ''
    NetworkTypes = []
    Metrics = []
    Throwaway = False

    @staticmethod
    def check_mismatching_parameters(test_configs, blueprint_configs):
        if type(test_configs) == list:
            missing_parameters = []
            obsolete_parameters = []
            for idx, (test_config, blueprint_config) in enumerate(zip(test_configs, blueprint_configs)):
                missing_parameters += [key + "({})".format(idx) for key, val
                                       in blueprint_config.items() if key not in test_config]
                obsolete_parameters += [key + "({})".format(idx) for key, val
                                        in test_config.items() if key not in blueprint_config]
        else:
            missing_parameters = [key for key, val in blueprint_configs.items() if key not in test_configs]
            obsolete_parameters = [key for key, val in test_configs.items() if key not in blueprint_configs]
        return missing_parameters, obsolete_parameters

    @staticmethod
    def validate_config(trainer_configuration,
                        agent_configuration,
                        exploration_configuration):
        missing_parameters, obsolete_parameters = \
            Agent.check_mismatching_parameters(trainer_configuration,
                                               agent_configuration.get('TrainingParameterSpace'))

        missing_net_parameters, obsolete_net_parameters = \
            Agent.check_mismatching_parameters(trainer_configuration.get('NetworkParameters'),
                                               agent_configuration.get('NetworkParameterSpace'))
        missing_expl_parameters, obsolete_expl_parameters = \
            Agent.check_mismatching_parameters(trainer_configuration.get('ExplorationParameters'),
                                               exploration_configuration.get('ParameterSpace'))

        wrong_type_parameters = \
            Agent.check_wrong_parameter_types(trainer_configuration,
                                              agent_configuration.get('TrainingParameterSpace'),
                                              obsolete_parameters)

        wrong_type_net_parameters = \
            Agent.check_wrong_parameter_types(trainer_configuration.get('NetworkParameters')[0],
                                              agent_configuration.get('NetworkParameterSpace')[0],
                                              obsolete_net_parameters)
        wrong_type_expl_parameters = \
            Agent.check_wrong_parameter_types(trainer_configuration.get('ExplorationParameters'),
                                              exploration_configuration.get('ParameterSpace'),
                                              obsolete_expl_parameters)

        return missing_parameters, obsolete_parameters, missing_net_parameters, obsolete_net_parameters, \
               missing_expl_parameters, obsolete_expl_parameters, wrong_type_parameters, wrong_type_net_parameters, \
               wrong_type_expl_parameters

    @staticmethod
    def validate_action_space(agent_configuration, environment_configuration):
        # Check for compatibility of environment and agent action space
        if environment_configuration.get("ActionType") not in agent_configuration.get("ActionType"):
            print("The action spaces of the environment and the agent are not compatible.")
            return False
        return True

    @staticmethod
    def check_wrong_parameter_types(test_config, blueprint_config, obsolete=[]):
        wrong_type_parameters = [" >> ".join([": ".join([key, str(val)]), str(blueprint_config.get(key))])
                                 for key, val in test_config.items()
                                 if type(val) != blueprint_config.get(key)
                                 and key not in obsolete]
        return wrong_type_parameters

    @staticmethod
    def get_config(config_dict):
        config_dict = {key: val for (key, val) in config_dict.items()
                       if not key.startswith('__')
                       and not callable(val)
                       and not type(val) is staticmethod
                       }
        return config_dict

    @staticmethod
    def get_dummy_action(agent_num, action_shape, action_type):
        if action_type == "CONTINUOUS":
            return np.random.random((agent_num, action_shape))
        else:
            return np.random.randint(0, action_shape, (agent_num, 1))

    def act(self, states, mode="training"):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def learn(self, replay_batch):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def build_network(self, network_parameters, environment_parameters):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def load_checkpoint(self, path):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def boost_exploration(self):
        return False

    @staticmethod
    def get_training_batch_from_replay_batch(replay_batch, observation_shapes, action_shape):
        state_batch = []
        next_state_batch = []
        for obs_shape in observation_shapes:
            state_batch.append(np.zeros((len(replay_batch), *obs_shape)))
            next_state_batch.append(np.zeros((len(replay_batch), *obs_shape)))
        try:
            action_batch = np.zeros((len(replay_batch), *action_shape))
        except TypeError:
            action_batch = np.zeros((len(replay_batch), action_shape))
        reward_batch = np.zeros((len(replay_batch), 1))
        done_batch = np.zeros((len(replay_batch), 1))

        for idx, transition in enumerate(replay_batch):
            for idx2, (state, next_state) in enumerate(zip(transition['state'], transition['next_state'])):
                state_batch[idx2][idx] = state
                next_state_batch[idx2][idx] = next_state
            action_batch[idx] = transition['action']
            reward_batch[idx] = transition['reward']
            done_batch[idx] = transition['done']
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
