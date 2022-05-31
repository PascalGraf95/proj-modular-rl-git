import sys
from types import ModuleType, FunctionType
from gc import get_referents
import numpy as np

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


def get_exploration_policies(num_policies: int = 1, beta_max: float = 0.3, gamma_max: float = 0.9999,
                             gamma_middle: float = 0.997, gamma_min: float = 0.99):
    """Calculate exploration policies consisting of (beta, gamma) pairs. Beta describes the intrinsic reward weighting
    in comparison to the basic environment reward and gamma the discount factor.

    Parameters
    ----------
    num_policies: int
        Total number of different exploration policy pairs (beta, gamma). Should ideally be minimum as large as the
        number of agents (Agent57 uses policy_num=32 with 256 actors.) -> With Meta-Learning number can be chosen
        freely as Bandit chooses the policy index at the beginning of each episode for each actor independently.
    beta_max: float
        Maximum beta value
    gamma_max: float
        Maximum gamma value
    gamma_middle: float
        Value after the maximum gamma value. Is Agent57 specific.
    gamma_min: float
        Minimum gamma value

    Returns
    ----------
    dict_exploration_policies: dict
        Dictionary containing the respective beta and gamma values.
        Data structure: {0: {'beta': _BETA_VALUE_MIN, 'gamma': _GAMMA_VALUE_MAX},
                         1: {'beta': _BETA_VALUE_AT_INDEX_1, 'gamma': _GAMMA_VALUE_AT_INDEX_1},
                         2: {'beta': _BETA_VALUE_AT_INDEX_2, 'gamma': _GAMMA_VALUE_AT_INDEX_2},
                         ...,
                         total number of policies: {'beta': _BETA_VALUE_MAX, 'gamma': _GAMMA_VALUE_MIN}}
    """

    assert num_policies > 0, "Please specify a policy number larger than 0."

    # Calculate exploration policy pairs (beta, gamma)
    dict_exploration_policies = {}
    if num_policies == 1:
        # Set policy values to min values if only one single exploration policy is used
        dict_exploration_policies[0] = {'beta': 0, 'gamma': gamma_max}
        return dict_exploration_policies
    else:
        # Otherwise iterate through policy indices and calculate respective betas and gammas
        for idx in range(num_policies):
            dict_exploration_policies[idx] = {'beta': get_beta(num_policies, beta_max, idx),
                                              'gamma': get_gamma(num_policies, gamma_max, gamma_middle, gamma_min, idx)}
        return dict_exploration_policies


def get_beta(num_policies, beta_max, policy_idx):
    """Calculate exploration policies consisting of (beta, gamma) pairs. Beta describes the intrinsic reward weighting
    in comparison to the basic environment reward and gamma the discount factor.

    Parameters
    ----------
    num_policies: int
        Total number of different exploration policy pairs (beta, gamma). Should ideally be minimum as large as the
        number of agents (Agent57 uses policy_num=32 with 256 actors.) -> With Meta-Learning number can be chosen
        freely as Bandit chooses the policy index at the beginning of each episode for each actor independently.
    beta_max: float
        Maximum beta value
    policy_idx: int
        Index of the current policy. A value between 0...Total number of policies.

    Returns
    ----------
    beta: float
        beta of the respective exploration policy.
    """

    if policy_idx == 0:
        beta = 0
    elif policy_idx == num_policies - 1:
        beta = beta_max
    else:
        beta = beta_max * sigmoid(10 * (2 * policy_idx - (num_policies - 2)) / (num_policies - 2))
    return beta


def get_gamma(num_policies, gamma_max, gamma_middle, gamma_min, policy_idx):
    """Calculate exploration policies consisting of (beta, gamma) pairs. Beta describes the intrinsic reward weighting
    in comparison to the basic environment reward and gamma the discount factor. Function differs for Agent57 and NGU,
    feel free to use the preferred method as both are implemented below. NGU-specific calculation results in a smoother
    and more general function, but aims for the same goal.

    Parameters
    ----------
    num_policies: int
        Total number of different exploration policy pairs (beta, gamma). Should ideally be minimum as large as the
        number of agents (Agent57 uses policy_num=32 with 256 actors.) -> With Meta-Learning number can be chosen
        freely as Bandit chooses the policy index at the beginning of each episode for each actor independently.
    gamma_max: float
        Maximum gamma value
    gamma_middle: float
        Value after the maximum gamma value. Is Agent57 specific.
    gamma_min: float
        Minimum gamma value
    policy_idx: int
        Index of the current policy. A value between 0...Total number of policies.

    Returns
    ----------
    gamma: float
        gamma of the respective exploration policy.
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
    return gamma


# Just for Debugging purposes
def print_policy_functions(dict_exploration_policies: dict):
    import matplotlib.pyplot as plt
    # print_policy_functions(get_exploration_policies(32))

    betas = []
    gammas = []

    num_policies = np.linspace(0, len(dict_exploration_policies) - 1, num=len(dict_exploration_policies))

    for idx in range(len(dict_exploration_policies)):
        betas.append(dict_exploration_policies[idx]['beta'])
        gammas.append(dict_exploration_policies[idx]['gamma'])

    plt.plot(num_policies, betas)
    plt.xlabel("policy")
    plt.ylabel("betas")
    plt.show()

    plt.plot(num_policies, gammas)
    plt.xlabel("policy")
    plt.ylabel("gammas")
    plt.show()


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig
