import sys
from types import ModuleType, FunctionType
from gc import get_referents
import numpy as np
import tensorflow as tf

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def get_exploration_policies(num_policies: int = 1, mode: str = "training", beta_max: float = 0.3,
                             gamma_max: float = 0.9999, gamma_middle: float = 0.997, gamma_min: float = 0.99):
    """Calculate exploration policies consisting of (beta, gamma) pairs. Beta describes the intrinsic reward weighting
    in comparison to the basic environment reward and gamma describes the discount factor. Beta is modeled as sigmoid
    function between 0 and beta max, gamma is modeled as negative exponential function between gamma_max and gamma_min.

    Parameters
    ----------
    num_policies: int
        Total number of different exploration policy pairs (beta, gamma). Should ideally be as large as the number of
        agents at minimum (Agent57 uses policy_num=32 with 256 actors.) -> With Meta-Learning number can be chosen
        freely as Bandit chooses the policy index at the beginning of each episode for each actor independently.
    beta_max: float
        Maximum beta value
    gamma_max: float
        Maximum gamma value
    gamma_middle: float
        Value after the maximum gamma value.
    gamma_min: float
        Minimum gamma value

    Returns
    -------
    dict_exploration_policies: dict
        Dictionary containing the respective beta and gamma values.
        Data structure: [0: {'beta': _BETA_VALUE_MIN, 'gamma': _GAMMA_VALUE_MAX},
                         1: {'beta': _BETA_VALUE_AT_INDEX_1, 'gamma': _GAMMA_VALUE_AT_INDEX_1},
                         2: {'beta': _BETA_VALUE_AT_INDEX_2, 'gamma': _GAMMA_VALUE_AT_INDEX_2},
                         ...,
                         total number of policies: {'beta': _BETA_VALUE_MAX, 'gamma': _GAMMA_VALUE_MIN}]
    """

    assert num_policies > 0, "Please specify a policy number larger than 0."

    # Calculate exploration policy pairs (beta, gamma)
    dict_exploration_policies = []
    if num_policies == 1:
        if mode == "training":
            # Set policy values to min values if only one single exploration policy is used
            dict_exploration_policies.append({'beta': 0, 'gamma': gamma_max, 'scaling': 1})
        else:
            # Set policy values to min values if only one single exploration policy is used
            dict_exploration_policies.append({'beta': 0, 'gamma': gamma_max, 'scaling': 0})
        return dict_exploration_policies
    else:
        scaling_distribution = np.linspace(0, 1, num_policies).tolist()
        # Otherwise, iterate through policy indices and calculate respective betas and gammas
        for idx in range(num_policies):
            dict_exploration_policies.append({'beta': get_beta(num_policies, beta_max, idx),
                                              'gamma': get_gamma(num_policies, gamma_max, gamma_middle,
                                                                 gamma_min, idx),
                                              'scaling': scaling_distribution[idx]})
        return dict_exploration_policies


def get_beta(num_policies, beta_max, policy_idx):
    """Calculate exploration policies' beta values.

    Parameters
    ----------
    num_policies: int
        Total number of different exploration policy pairs (beta, gamma).
    beta_max: float
        Maximum beta value
    policy_idx: int
        Index of the current policy. A value between 0...Total number of policies.

    Returns
    -------
    beta: float
        beta values
    """

    if policy_idx == 0:
        beta = 0
    elif policy_idx == num_policies - 1:
        beta = beta_max
    else:
        beta = beta_max * sigmoid(10 * (2 * policy_idx - (num_policies - 2)) / (num_policies - 2))
    return float(beta)


def get_gamma(num_policies, gamma_max, gamma_middle, gamma_min, policy_idx):
    """Calculate exploration policies' gamma values. Function differs for Agent57 and NGU,
    feel free to use the preferred method as both are implemented below. NGU-specific calculation results in a smoother
    and more general function, but aims for the same goal.

    Parameters
    ----------
    num_policies: int
        Total number of different exploration policy pairs (beta, gamma)
    gamma_max: float
        Maximum gamma value
    gamma_middle: float
        Value after the maximum gamma value.
    gamma_min: float
        Minimum gamma value
    policy_idx: int
        Index of the current policy. A value between 0...Total number of policies.

    Returns
    -------
    gamma: float
        gamma values
    """

    # region Never-Give-Up
    if policy_idx == 0:
        gamma = gamma_max
    elif policy_idx == num_policies - 1:
        gamma = gamma_min
    else:
        gamma = ((num_policies - 1 - policy_idx) * np.log(1 - gamma_max) +
                 policy_idx * np.log(1 - gamma_min)) / (num_policies - 1)
        gamma = 1 - np.exp(gamma)
    # endregion

    # region Agent57
    # Does not give the same result as presented within the paper (https://arxiv.org/pdf/2003.13350.pdf).
    # -> Fault within the paper?
    '''
    if policy_idx == 0:
        gamma = gamma_max
    elif (policy_idx >= 1) and (policy_idx <= 6):
        gamma = gamma_middle + (gamma_max - gamma_middle) * sigmoid(10 * ((2 * policy_idx - 6) / 6))
    elif policy_idx == 7:
        gamma = gamma_middle
    else:
        gamma = ((num_policies - 9) * np.log(1 - gamma_middle) +
                 (policy_idx - 8) * np.log(1 - gamma_min)) / (num_policies - 9)
        gamma = 1 - np.exp(gamma)
    '''
    # endregion
    return float(gamma)


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def modify_observation_shapes(observation_shapes, action_shapes, action_type, feedback_actions, feedback_rewards,
                              feedback_exploration_policy):
    """Extend current observation shapes through additional shape values. Those additional shapes will be used to
    feed inputs that are not sent through the interface (Unity, Gym) but are generated by the agent itself. Such inputs
    would be for example the prior action of the agent and the prior extrinsic reward ('prior' means the prior
    environment step).

    Parameters
    ----------
    observation_shapes:
        Observation shapes given through the interface (Unity, Gym)
    action_shapes:
        Action shapes given through the interface (Unity, Gym)
    action_type:
        Action type given through the interface
    feedback_actions:
        Boolean whether or not to feed back the last step action as a network input.
    feedback_rewards:
        Boolean whether or not to feed back the last step rewards (extr. and intr.) as a network input.
    feedback_exploration_policy:
        Boolean whether or not to feed back the last step exploration_policy as a network input.

    Returns
    -------
    modified_observation_shapes:
        Extended version of the original observation shapes.
    """
    modified_observation_shapes = []
    for obs_shape in observation_shapes:
        modified_observation_shapes.append(obs_shape)

    # Prior action
    if feedback_actions:
        if action_type == "CONTINUOUS":
            modified_observation_shapes.append((action_shapes,))
        elif action_type == "DISCRETE":
            modified_observation_shapes.append((1,))
        else:
            print("Used action type not implemented yet.")
    # Prior rewards
    if feedback_rewards:
        # extrinsic
        modified_observation_shapes.append((1,))
        # intrinsic
        modified_observation_shapes.append((1,))

    # Prior j (index of exploration policy (beta, gamma))
    if feedback_exploration_policy:
        modified_observation_shapes.append((1,))

    return modified_observation_shapes


def set_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
