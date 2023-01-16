# >>> Modular Reinforcement Learning©
*created and maintained by Pascal Graf 2022*

**Documentation and instructions are incorporated into the code. To get started take a look at 
the `main.py` file.**

## Package Versions:
*The following packages have successfully been used with this framework. Other versions might cause errors.*

- ### CUDA
  - **Option 1**:
    - Cuda Version: 10.1.X (in combination with tensorflow 2.3.X)
    - CuDnn Version: 7.6.X (corresponds to the Cuda version, see: https://www.tensorflow.org/install/source)
  - **Option 2 (Updated 17.10.2022):** 
    - Cuda Version: 11.8.X (in combination with tensorflow 2.10.X)
    - CuDnn Version: 8.6.X (corresponds to the Cuda version, see: https://www.tensorflow.org/install/source)

*Warning: Utilizing Option 2 with Windows 10 might lead to occasional 'Optimization loop failed' warning messages 
which do not seem to affect performance.*
- ### PIP
  - ray: 1.11.X -> NOTE: 2.0.X also works and enables debugging (set `ray.init(local_mode=True)`). 
However, on Windows 10 this might throw 'Windows fatal exception' errors that DO NOT affect performance or executability.
  - **Option 1**:
    - tensorflow: 2.3.X
    - tensorflow-probability: 0.11.X (corresponds to tensorflow version)
  - **Option 2**:
    - tensorflow: 2.10.X
    - tensorflow-probability: 0.11.X (corresponds to tensorflow version)
## Features
- ###Interfaces:
  - Unity MlAgentsV18
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
  - Curriculum Learning (restricted)
- ###Exploration Algorithms:
  - Epsilon Greedy
  - Intrinsic Curiosity Module (ICM)
  - Random Network Distillation (RND)

## To Do:
- Incorporate CARLA interface into the main branch
- Implement Offline Reinforcement Learning methods
- SAC for discrete action spaces ([Paper](https://arxiv.org/pdf/1910.07207.pdf), 
[Blog](https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a))
- Incorporate Attention architecture into the main branch
- Fix `Intrinsic Curiosity Module (ICM)`
- Implement Meta Learning and "Never Give Up" (Agent57)
- Extend self play functionality
- Implement Multi Agent RL algorithms for cooperative / competitive learning
- Implement additional Imitation Learning methods