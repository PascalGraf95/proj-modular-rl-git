import numpy as np
from collections import deque
from sklearn import mixture
from .exploration_algorithm_blueprint import ExplorationAlgorithm


class PseudoCount(ExplorationAlgorithm):
    Name = "PseudoCount"
    ActionAltering = False
    IntrinsicReward = True

    ParameterSpace = {
        'ObservationBinSize': float,
        'ComponentsGMM': float,
        'SlopeScale': float,
        'MaxIterationsGMM': float
    }

    def __init__(self,  action_shape, action_space, parameters):
        self.observation_bin_size = parameters["ObservationBinSize"]  # 50_000
        self.components_gmm = parameters["ComponentsGMM"]  # 20
        self.slope_scale = parameters["SlopeScale"]  # 1
        self.max_iter = parameters["MaxIterationsGMM"]  # 500

        self.gaussian_model = mixture.BayesianGaussianMixture(n_components=self.components_gmm, max_iter=self.max_iter)
        # self.weights = []
        self.observation_bin = deque(maxlen=self.observation_bin_size)


    @staticmethod
    def get_config():
        config_dict = PseudoCount.__dict__
        return ExplorationAlgorithm.get_config(config_dict)


    def get_pseudo_count(self, state):
        if len(self.observation_bin) > self.components_gmm and len(self.observation_bin) > 1000:
            # fit model p to all states that have been seen so far and calculate p(s)
            data = np.array(self.observation_bin)  # transform deque to array with (n_samples, n_features)
            p_theta = self.gaussian_model.fit(data)
            # print(self.gaussian_model.weights_)
            # self.weights = self.gaussian_model.weights_
            state = np.expand_dims(state, axis=0)  # score needs shape (1, n_features)
            p_s = 10 ** (p_theta.score(state))

            # store state in bin, fit new model p' to (Data u state) and calculate p'(s)
            self.observation_bin.append(state[0])
            data = np.array(self.observation_bin)
            p_theta = self.gaussian_model.fit(data)
            p_dash_s = 10 ** (p_theta.score(state))

            # use p(s) and p'(s) to estimate pseudo_count
            pseudo_count = p_s * (1 - p_dash_s) / (p_dash_s - p_s)
            return pseudo_count
        return None

    @staticmethod
    def act(*args):
        return None


    def get_intrinsic_reward(self, decision_steps):
        state = decision_steps.obs
        # check if there are any states in decision_steps
        if state[0].shape[0] == 0:
            return 0

        # iterate over de states (multiple agents -> multiple states)
        intrinsic_reward_list = []
        for s in state[0]:
            self.observation_bin.append(s)
            pseudo_count = self.get_pseudo_count(s)
            if pseudo_count:
                if pseudo_count > 0:
                    intrinsic_reward_list.append(self.slope_scale * (pseudo_count ** -0.5))
                else:
                    intrinsic_reward_list.append(0)

        # only return intrinsic reward list if every state got a reward
        if len(intrinsic_reward_list) == state[0].shape[0]:
            return intrinsic_reward_list
        return 0
