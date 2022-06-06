# >>> Modular Reinforcement Learning©
*created and maintained by Pascal Graf 2022*

**Documentation and instructions are incorporated into the code. To get started take a look at 
the `main.py` file.**

## Package Versions:
*The following packages have successfully been used with this framework. Other versions might cause errors.*
- ### CUDA
  - Cuda Version: 10.1.X (Later versions, especially 11.X lead to memory leaks on Windows machines)
  - CuDnn Version: 7.6.X (corresponds to the Cuda version, see: https://www.tensorflow.org/install/source)
- ### PIP
  - tensorflow: 2.3.X
  - tensorflow-probability: 0.11.X (corresponds to tensorflow version)
  - ray: 1.11.X (later versions, especially 1.12 lead to Windows fatal errors 'access violation')
## Features
- ###Interfaces:
  - Unity MlAgentsV18
  - OpenAI Gym
- ###Learning Algorithms:
  - **Deep Q Learning / Deep Q Network (DQN)** with several extensions
  (Double Learning, Dueling Architecture, Noisy Networks)
  - **Soft Actor-Critic (SAC)** with automatic temperature parameter adjustment
- ###Learning Features:
  - Several network architectures for vector and image-based environments 
  - Recurrent Neural Networks (R2D2) with burn in
  - Prioritized Experience Replay (PER)
  - Potential feature preprocessing
  - Curriculum Learning
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
- Fix and extend self play functionality
- Implement Multi Agent RL algorithms for cooperative / competitive learning
- Implement Imitation Learning methods