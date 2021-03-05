#!/usr/bin/env python

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel


class MlAgentsV12Interface:
    @staticmethod
    def get_behavior_name(env: UnityEnvironment):
        spec = env.behavior_specs
        assert len(spec.keys()) == 1, "This trainer currently only supports environment with one behavior."
        for b in env.behavior_specs:
            return b

    @staticmethod
    def get_observation_shapes(env: UnityEnvironment):
        spec = env.behavior_specs
        behavior_spec = [v for v in spec.values()][0]
        return behavior_spec.observation_shapes

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
        MlAgentsV12Interface.reset(env)
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
    def get_transitions(env: UnityEnvironment,
                        environment_configuration: dict,
                        actions,
                        last_transitions: dict = {}):
        decision_steps, terminal_steps = env.get_steps(environment_configuration['BehaviorName'])
        transitions = {'states': {}, 'actions': {}, 'rewards': {}, 'next_states': {},
                       'changed_indices': [], 'done_indices': [], 'need_action_indices': decision_steps.agent_id}
        remaining_agent_indices = list(range(environment_configuration['AgentNumber']))
        # Catch Decision Steps
        for idx, agent_id in enumerate(decision_steps.agent_id):
            decision_step = decision_steps[agent_id]
            remaining_agent_indices.remove(agent_id)
            # Build new transition from old transition, last action and new observations.
            try:
                transitions['states'][agent_id] = last_transitions.get('next_states')[agent_id]
            except TypeError:
                transitions['states'][agent_id] = decision_step.obs
            transitions['actions'][agent_id] = actions[idx]
            transitions['rewards'][agent_id] = decision_step.reward
            transitions['next_states'][agent_id] = decision_step.obs

            # Only add Agent ID to the changed indices if it was not in terminal steps last iteration.
            if last_transitions.get('done_indices'):
                if agent_id not in last_transitions.get('done_indices'):
                    transitions['changed_indices'].append(agent_id)

        # Catch Terminal Steps
        for idx, agent_id in enumerate(terminal_steps.agent_id):
            terminal_step = terminal_steps[agent_id]

            remaining_agent_indices.remove(agent_id)
            # Build new transition from old transition, last action and new observations.
            try:
                transitions['states'][agent_id] = last_transitions.get('next_states')[agent_id]
            except TypeError:
                transitions['states'][agent_id] = terminal_step.obs
            transitions['actions'][agent_id] = actions[idx]
            transitions['rewards'][agent_id] = terminal_step.reward
            transitions['next_states'][agent_id] = terminal_step.obs
            # Append to Changed and Done Indices
            transitions['changed_indices'].append(agent_id)
            transitions['done_indices'].append(agent_id)

        # Catch Remaining Agents
        for idx in remaining_agent_indices:
            # Copy the last transition
            transitions['states'][idx] = last_transitions.get('states')[idx]
            transitions['actions'][idx] = last_transitions.get('actions')[idx]
            transitions['rewards'][idx] = last_transitions.get('rewards')[idx]
            transitions['next_states'][idx] = last_transitions.get('next_states')[idx]
            if idx in last_transitions.get('done_indices'):
                transitions['done_indices'].append(idx)
        return transitions

    @staticmethod
    def step_action(env: UnityEnvironment, behavior_name: str, actions):
        if actions.shape[0] > 0:
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
        env.step()
