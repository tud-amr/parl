# PARL: Predictability-Aware Reinforcement Learning
Paper: https://arxiv.org/abs/2311.18703

Authors: Daniel Jarne Ornia*, Giannis Delimpaltadakis*, Jens Kober and Javier Alonso-Mora

## Training PARL agents
We implemented PARL based on Stable Baselines3 agents.
To train PARL agents with the paper's pre-tuned hyperparameters and reproduce the results, simpy run

`python train_parl_agents.py env_name steps k seed`

Example:

`python train_parl_agents.py HalfCheetah-v4 1000000 0.5 0`

The module follows the syntax of Stable Baselines3, with additional parameters required for the model learning and entropy rate estimations described in the class code. The code is written to work seamlessly as any other SB3 algorithm. To train an agent following stantard SB3 syntax, see file train_sb3_parl.py.

### Hyperparameters
The file parameters_parl.yml contains the tuned hyperparameters used in the paper.

### Trained agents
We provide the trained agents used for all results presented in the paper in [the following link](https://surfdrive.surf.nl/files/index.php/s/lFmrat9FpUbBrzR).

## Implementing new predictability-aware algorithms
PARL is designed to be applicable to any on and off policy RL algorithm. The current version is based on PPO, but the base agent class in /src/agent/BaseAgents.py implements a general on-policy model based RL class, in principle adaptable to other on policy algorithms (with minimum modifications).

### Dynamic Models
The dynamic model classes are under /src/utils/dynamic_model.py. Different architectures can be defined in the same file and swapped into the algorithm parameters, but it has not been tested with other types of models.


## Conda Env

We provide a yml file with the environment dependencies. You can install it via:
`conda env create -f parl.yml`

and activate it:

`conda activate parl`

## Grid robotic tasks
We provide the code for the grid environments including the moving obstacle switch task and slippery navigation presented in the Appendix of the paper. 
