"""Abstract base classes for RL algorithms."""

import io
import sys
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from copy import deepcopy
from src.utils.dynamic_model import DynamicModel,DynamicModelEncoder, DynamicModelEnv
from src.utils.buffers import RolloutModelBasedBuffer, RolloutBufferComplexity
from src.utils.wrappers import simplifier_static,simplifier_agent

import numpy as np
import torch as th
from torch.nn.functional import mse_loss
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import (
    should_collect_more_steps,
    obs_as_tensor
)
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecNormalize,
)

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")
SelfOnPolicyModelBasedAlgorithm = TypeVar("SelfOnPolicyModelBasedAlgorithm", bound="OnPolicyModelBasedAlgorithm")
SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")
SelfOffPolicyModelBasedAlgorithm = TypeVar("SelfOffPolicyModelBasedAlgorithm", bound="OffPolicyModelBasedAlgorithm")


def maybe_make_env(env: Union[GymEnv, str, None], verbose: int) -> Optional[GymEnv]:
    """If env is a string, make the environment; otherwise, return env.

    :param env: The environment to learn from.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating if envrironment is created
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        if verbose >= 1:
            print(f"Creating environment from the given name '{env}'")
        env = gym.make(env)
    return env


class OnPolicyModelBasedAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param model_kwargs: additional arguments to be passed to the model on creation
    :param model_steps: Number of steps to run the model for each environment per update
    :param model_grad_steps: Number of gradient steps to run on the model
    :param model_buffer_size: Size of the replay buffer for the model
    :param unnormalize_model: Whether or not to unnormalize the model inputs
    :param clip_likelihood: Likelihood clipping value
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param norm_parameter: Normalization parameter for the model
    :param pre_train_steps: Number of steps to pre-train the model
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBufferComplexity
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_steps: int=32,
        model_grad_steps: int=1,
        model_buffer_size: int=5000000,
        unnormalize_model: bool = False,
        clip_likelihood: float = 1.0,
        verbose: int = 0,
        norm_parameter: float = 1.0,
        pre_train_steps: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        self.model_kwargs = model_kwargs or {}
        self.model_steps = model_steps
        self.model_grad_steps = model_grad_steps
        self.num_timesteps_m = 0
        self._last_obs_m = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self.n_steps = n_steps
        self.model_buffer_size = model_buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self._n_updates_model = 0
        self.pre_train_steps = pre_train_steps
        self.clip_likelihood = clip_likelihood
        self.unnormalize_model = unnormalize_model
        self.unnormalized_states = None
        self.ep_rewards = []
        self.ep_len = []
        self.rew = np.asarray([0.0 for i in range(self.n_envs)])
        self.norm_parameter = norm_parameter
        self.complex_rate = 0.0
        self.true_complex_rate = 0.0
        self.max_complexity = 1
        self.min_complexity = -2
        self.runningmean = RunningMeanStd()
        self.simplify_obs = lambda x: x #TODO: Remove, legacy

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBufferComplexity
        obs_space_reduced = self.observation_space
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            obs_space_reduced=obs_space_reduced,
        )
        self.replay_buffer_model = ReplayBuffer(
            self.model_buffer_size,
            obs_space_reduced,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            handle_timeout_termination=False
        )
        self.model_rollout_buffer = buffer_cls(
            self.model_steps,
            obs_space_reduced,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        feature_extractor = self.policy.features_extractor_class
        if feature_extractor is None:
            feature_extractor = 'mlp'
        type = "mlp"
        if "type" in self.model_kwargs.keys():
            type = self.model_kwargs["type"]
        if type == "autoencoder":
            self.dynamic_model = DynamicModelEncoder(self.env, feature_extraction=feature_extractor,
                                              device=self.device, **self.model_kwargs)
        else:
            self.dynamic_model = DynamicModel(self.env, feature_extraction=feature_extractor,
                                              device=self.device, **self.model_kwargs)
        if isinstance(self.env, VecEnv):
            self.simulator = DynamicModelEnv(self.env, self.dynamic_model, vectorized=True)
        else:
            self.simulator = DynamicModelEnv(self.env, self.dynamic_model, vectorized=False)
        self.policy = self.policy.to(self.device)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        a,b = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self._last_obs_m = self.simulator.reset()[0]
        return a,b

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBufferComplexity,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        if isinstance(env, VecNormalize):
            last_obs_norm = self.simplify_obs(env.get_original_obs())/self.norm_parameter
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        last = 0
        bootstrap_complexities = np.zeros_like(rollout_buffer.rewards)
        complx = 0

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                complex_values = self.policy.predict_complex_values(obs_tensor)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            # slow update of complexity rate

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.rew += rewards
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if done:
                    self.ep_rewards.append(self.rew[idx])
                    self.rew[idx] = 0
                    self.ep_len.append(n_steps-last)
                    last = n_steps
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        terminal_complex_value = self.policy.predict_complex_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
                    bootstrap_complexities[n_steps-1,idx] += terminal_complex_value

            if self.unnormalize_model:
                new_obs_norm = self.simplify_obs(env.get_original_obs()) / self.norm_parameter
                last_obs_2 = last_obs_norm
            else:
                new_obs_norm = new_obs
                last_obs_2 = self._last_obs

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                new_obs,
                actions,
                rewards,
                dones,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                last_obs_2,
                new_obs_norm,
                complexity_value=complex_values,
            )
            self.replay_buffer_model.add(
                last_obs_2,  # type: ignore[arg-type]
                new_obs_norm,
                actions,
                rewards,
                dones,
                self._last_episode_starts,  # type: ignore[arg-type]
            )
            last_obs_norm = new_obs_norm
            self.rew += np.mean(rewards)
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        if self.dynamic_model.has_encoder:
            with th.no_grad():
                _,latent,means = self.dynamic_model(
                    th.as_tensor(rollout_buffer.swap_and_flatten(rollout_buffer.observations_no_norm)).to(self.dynamic_model.device),
                    th.as_tensor(rollout_buffer.swap_and_flatten(rollout_buffer.actions)).to(self.dynamic_model.device),
                th.as_tensor(rollout_buffer.swap_and_flatten(rollout_buffer.next_observations_no_norm)).to(self.dynamic_model.device))
                latent = latent.cpu().numpy()
                means = means.cpu().numpy()
        else:
            with th.no_grad():
                latent = None
                means = self.dynamic_model.predict_mean(
                    th.as_tensor(rollout_buffer.observations_no_norm).to(self.dynamic_model.device),
                    th.as_tensor(rollout_buffer.actions).to(self.dynamic_model.device)).cpu().numpy()
        rollout_buffer.compute_complexities(means,bootstrap_complexities,latent)
        self.true_complex_rate = (1 - 0.1) * self.true_complex_rate + \
                                 0.1 * rollout_buffer.mean_complexity

        self.complex_rate = (1 - self.complex_rate_learning_rate) * self.complex_rate + \
                            self.complex_rate_learning_rate * rollout_buffer.mean_complexity
        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            complexity_values = self.policy.predict_complex_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones,
                                                     last_complex_values=complexity_values, rate=self.complex_rate)

        callback.on_rollout_end()

        if self.ep_rewards == []:
            self.ep_rewards.append(np.mean(self.rew))
            self.ep_len.append(n_rollout_steps)
        # self.rew *= 0
        return True

    def collect_rollouts_model(
        self,
        env: DynamicModelEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutModelBasedBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs_m is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.dynamic_model.train(False)

        n_steps_m = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        while n_steps_m < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps_m % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor_m = obs_as_tensor(self._last_obs_m, self.device)
                actions_m, values_m, log_probs_m = self.policy(obs_tensor_m)
            actions_m = actions_m.cpu().numpy()

            # Rescale and perform action
            clipped_actions_m = actions_m
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions_m = np.clip(actions_m, self.action_space.low, self.action_space.high)

            new_obs_m, rewards_m, dones_m, infos_m = env.step(clipped_actions_m)

            self.num_timesteps_m += env.num_envs

            # self._update_info_buffer(infos_m)
            n_steps_m += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions_m = actions_m.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones_m):
                if (
                    done
                    and infos_m[idx].get("terminal_observation") is not None
                    and infos_m[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos_m[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards_m[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs_m,  # type: ignore[arg-type]
                new_obs_m,
                actions_m,
                rewards_m,
                dones_m,
                self._last_episode_starts,  # type: ignore[arg-type]
                values_m,
                log_probs_m,
            )
            self._last_obs_m = new_obs_m  # type: ignore[assignment]
            self._last_episode_starts = dones_m

        with th.no_grad():
            # Compute value for the last timestep
            values_m = self.policy.predict_values(obs_as_tensor(new_obs_m, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values_m, dones=dones_m)

        return True

    def train_model(self,grad_steps=None, pre_training=False) -> None:
        """
        Train the model for one gradient step on the sampled data.
        """
        self.dynamic_model.train()
        encoder_losses = []
        mean_losses = []
        if grad_steps is None:
            grad_steps = self.model_grad_steps

        if isinstance(self.dynamic_model,DynamicModel):
            for _ in range(grad_steps):
                replay_data = self.replay_buffer_model.sample(self.model_steps)
                log_likelihood_loss = 0
                obs = replay_data.observations.float().to(self.device)
                next_obs = replay_data.next_observations.float().to(self.device)
                acts = replay_data.actions.float().to(self.device)
                obs_mean = self.dynamic_model.predict_mean(obs, acts)
                loss_means = th.nn.functional.l1_loss(obs_mean,next_obs, reduction='mean')

                self.dynamic_model.optimizer.zero_grad()
                loss_means.backward()
                self.dynamic_model.optimizer.step()

                self._n_updates_model += 1
                mean_losses.append(loss_means.item())
        else:
            # Model is autoencoder-based
            for _ in range(grad_steps):
                replay_data = self.replay_buffer_model.sample(self.model_steps)
                # for replay_data in replay_generator:
                obs = replay_data.observations.float().to(self.device)
                next_obs = replay_data.next_observations.float().to(self.device)
                acts = replay_data.actions.float().to(self.device)
                x_hat, y_hat, y_mean = self.dynamic_model(obs, acts, next_obs)
                loss_encoder = th.nn.functional.smooth_l1_loss(x_hat,obs, reduction='mean')
                loss_model_mean = th.nn.functional.smooth_l1_loss(y_hat,y_mean, reduction='mean')
                loss_states = 2*loss_encoder + loss_model_mean
                self.dynamic_model.optimizer.zero_grad()
                loss_states.backward()
                self.dynamic_model.optimizer.step()
                self._n_updates_model += 1
                mean_losses.append(loss_states.item())
                encoder_losses.append(loss_encoder.item())
        if self.dynamic_model.decay_rate:
            progress = self.num_timesteps / self._total_timesteps
            for param_group in self.dynamic_model.optimizer.param_groups:
                param_group['lr'] = self.dynamic_model.learning_rate* (1 - progress)
        if grad_steps>0 and not pre_training:
            self.logger.record("train/n_updates_model", self._n_updates_model)
            self.logger.record("train/loss_model_mean", np.mean(mean_losses))
            self.logger.record("train/loss_model_encoder", np.mean(encoder_losses))
        self.dynamic_model.eval()

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyModelBasedAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyModelBasedAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        for i in range(self.pre_train_steps):
            self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            self.num_timesteps = 0
        self.train_model(grad_steps=self.pre_train_steps, pre_training=True)
        self._last_obs = self.env.reset()
        lens = 0
        rews = 0

        while self.num_timesteps < total_timesteps:
            self.ep_rewards = []
            self.ep_len = []
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            if continue_training is False:
                break
            self.train_model()
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                else:
                    rews = rews*0.97 + 0.03*safe_mean(self.ep_rewards)
                    lens = lens*0.97 + 0.03*safe_mean(self.ep_len)
                    self.logger.record("rollout/ep_rew_mean", rews)
                    self.logger.record("rollout/ep_len_mean", lens)
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
            self.train()

        callback.on_training_end()
        self.last_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        self.last_length = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
