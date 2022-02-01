#!/usr/bin/env python

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import time


class MlAgentsV18Interface:
    """
    This interface translates the universal commands used in this repository for environment communication to
    interact with Unity environments which have installed Ml-AgentsV18.
    """
    @staticmethod
    def get_behavior_name(env: UnityEnvironment):
        behavior_names = []
        behavior_clone_names = []
        for b in env.behavior_specs:
            if "Clone" in b:
                behavior_clone_names.append(b)
            else:
                behavior_names.append(b)
        assert len(behavior_clone_names) <= 1, "This trainer currently only supports environments with one behavior" \
                                               "(+ one clone of that behavior for self play)"
        assert len(behavior_names) == 1, "This trainer currently only supports environments with one behavior" \
                                         "(+ one clone of that behavior for self play)"
        if len(behavior_clone_names):
            return behavior_names[0], behavior_clone_names[0]
        return behavior_names[0], None

    @staticmethod
    def get_observation_shapes(env: UnityEnvironment):
        spec = env.behavior_specs
        behavior_spec = [v for v in spec.values()][0].observation_specs
        observation_shape = [b.shape for b in behavior_spec]
        return observation_shape

    @staticmethod
    def get_random_action(env: UnityEnvironment, agent_number):
        spec = env.behavior_specs
        behavior_spec = [v for v in spec.values()][0]
        return behavior_spec.action_spec.random_action(agent_number)

    @staticmethod
    def get_action_shape(env: UnityEnvironment, action_type):
        spec = env.behavior_specs
        behavior_spec = [v for v in spec.values()][0]
        if action_type == "CONTINUOUS":
            return behavior_spec.action_spec.continuous_size
        if action_type == "DISCRETE":
            return behavior_spec.action_spec.discrete_size

    @staticmethod
    def get_action_type(env: UnityEnvironment):
        spec = env.behavior_specs
        behavior_spec = [v for v in spec.values()][0]
        if behavior_spec.action_spec.is_continuous():
            return "CONTINUOUS"
        elif behavior_spec.action_spec.is_discrete():
            return "DISCRETE"

    @staticmethod
    def get_agent_number(env: UnityEnvironment, behavior_name: str):
        MlAgentsV18Interface.reset(env)
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_id_offset = np.min(decision_steps.agent_id)
        return len(decision_steps), agent_id_offset

    @staticmethod
    def get_steps(env: UnityEnvironment, behavior_name: str):
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        return decision_steps, terminal_steps

    @staticmethod
    def reset(env: UnityEnvironment):
        env.reset()

    @staticmethod
    def step_action(env: UnityEnvironment, action_type: str,
                    behavior_name: str,
                    actions,
                    behavior_clone_name=None,
                    clone_actions=None):

        if actions.shape[0] > 0:
            if action_type == "CONTINUOUS":
                env.set_actions(behavior_name, ActionTuple(continuous=actions))
            elif action_type == "DISCRETE":
                env.set_actions(behavior_name, ActionTuple(discrete=actions))
        if np.any(clone_actions):
            if action_type == "CONTINUOUS":
                env.set_actions(behavior_clone_name, ActionTuple(continuous=clone_actions))
            elif action_type == "DISCRETE":
                env.set_actions(behavior_clone_name, ActionTuple(discrete=clone_actions))
        env.step()
