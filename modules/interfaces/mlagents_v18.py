#!/usr/bin/env python

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel


class MlAgentsV18Interface:
    """
    This interface translates the universal commands used in this repository for environment communication to
    interact with Unity environments which have installed Ml-AgentsV18.
    """
    @staticmethod
    def get_behavior_name(env: UnityEnvironment):
        spec = env.behavior_specs
        assert len(spec.keys()) == 1, "This trainer currently only supports environments with one behavior."
        for b in env.behavior_specs:
            return b

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
    def get_action_shape(env: UnityEnvironment):
        spec = env.behavior_specs
        behavior_spec = [v for v in spec.values()][0]
        return behavior_spec.action_spec.continuous_size

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
        return len(decision_steps)

    @staticmethod
    def get_steps(env: UnityEnvironment, behavior_name: str):
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        return decision_steps, terminal_steps

    @staticmethod
    def reset(env: UnityEnvironment):
        env.reset()

    @staticmethod
    def step_action(env: UnityEnvironment, behavior_name: str, actions):
        if actions.shape[0] > 0:
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
        env.step()
