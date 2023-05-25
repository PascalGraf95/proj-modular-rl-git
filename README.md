# >>> Modular Reinforcement Learning©
*created and maintained by Pascal Graf 2023*

**Documentation and instructions are incorporated into the code. To get started take a look at 
the `main.py` file.**

## Package Versions:
*The following packages have successfully been used with this framework. Other versions might cause errors.*

- ### CUDA
- Cuda Version: 11.8.X (in combination with tensorflow 2.10.X)
- CuDnn Version: 8.6.X (corresponds to the Cuda version, see: https://www.tensorflow.org/install/source)

*Warning: Utilizing TF2.10.X with Windows 10 might lead to occasional 'Optimization loop failed' warning messages 
which do not seem to affect performance.*

- ### PIP
  - ray: 2.0.X enables debugging (set `ray.init(local_mode=True)`). 
However, on Windows 10 this might throw 'Windows fatal exception' errors that DO NOT affect performance or executability.
  - **Tensorflow**:
    - tensorflow: 2.10.X
    - tensorflow-probability: 0.18.X (corresponds to tensorflow version)
    
## Features
- ###Interfaces:
  - Unity MlAgentsV20
  - OpenAI Gym
- ###Learning Algorithms:
  - **Deep Q Learning / Deep Q Network (DQN)** with several extensions
  (Double Learning, Dueling Architecture, Noisy Networks)
  - **Soft Actor-Critic (SAC)** with automatic temperature parameter adjustment
  - **Conservative Q-Learning (CQL)** for offline Reinforcement Learning
- ###Learning Features:
  - Several network architectures for vector and image-based environments 
  - Recurrent Neural Networks (R2D2) with burn in
  - Prioritized Experience Replay (PER)
  - Potential feature preprocessing
  - Action & Reward Feedback
- ###Exploration Algorithms:
  - Epsilon Greedy
  - Intrinsic Curiosity Module (ICM)
  - Random Network Distillation (RND)
  - Never Give Up (NGU)
  - Episodic Novelty Module (ENM)

## To Do:
- SAC for discrete action spaces ([Paper](https://arxiv.org/pdf/1910.07207.pdf), 
[Blog](https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a))
- Incorporate Attention architecture into the main branch
- Fix `Intrinsic Curiosity Module (ICM)`
- Merge Meta Learning (Agent57)
- Extend self play functionality
- Implement Multi Agent RL algorithms for cooperative / competitive learning
- Implement additional Imitation Learning methods