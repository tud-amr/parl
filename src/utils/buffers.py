import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    observations_no_norm: th.Tensor
    next_observations_no_norm: th.Tensor
    complexities: th.Tensor
    complex_advantages: th.Tensor
    complex_returns: th.Tensor
    complex_values: th.Tensor


try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

#
class RolloutModelBasedBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    next_observations: np.ndarray
    observations_no_norm: np.ndarray
    next_observations_no_norm: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.observations_no_norm = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations_no_norm = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        :param last_complexity_values: complexity value estimation for the last step (one for each env)
        """
        # Convert to numpy


        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        obs_no_norm: np.ndarray = None,
        next_obs_no_norm: np.ndarray = None,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        if obs_no_norm is not None:
            self.observations_no_norm[self.pos] = np.array(obs_no_norm).copy()
            self.next_observations_no_norm[self.pos] = np.array(next_obs_no_norm).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "next_observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "observations_no_norm",
                "next_observations_no_norm"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.dones[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.episode_starts[batch_inds],
            self.observations_no_norm[batch_inds],
            self.next_observations_no_norm[batch_inds]
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


# TODO FIX BUFFER
class RolloutBufferComplexity(RolloutModelBasedBuffer):
    complexities: np.ndarray
    complex_advantages: np.ndarray
    complex_returns: np.ndarray
    complex_values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        obs_space_reduced: spaces.Space = None,
    ):
        if obs_space_reduced is None:
            self.obs_space_reduced = observation_space
        else:
            self.obs_space_reduced = obs_space_reduced
        self.obs_shape_reduced = get_obs_shape(self.obs_space_reduced)
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.mean_complexity = 0
        self.generator_ready = False
        self.reset()
        self.observations_no_norm = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape_reduced), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations_no_norm = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape_reduced), dtype=np.float32)


    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.complexities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.complex_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.complex_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.complex_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()
        self.observations_no_norm = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape_reduced), dtype=np.float32)
        self.next_observations_no_norm = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape_reduced),
                                                  dtype=np.float32)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray, last_complex_values: th.tensor,
                                      rate) -> None:
        lam = 0.1
        last_gae_lam_c = 0
        last_gae_lam = 0
        last_values = last_values.clone().cpu().numpy().flatten()
        last_complex_values = last_complex_values.clone().cpu().numpy().flatten()
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
                next_values_complex = last_complex_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_values_complex = self.complex_values[step + 1]
            delta_c = self.complexities[step]- rate + next_values_complex * next_non_terminal - self.complex_values[step]
            last_gae_lam_c = delta_c + lam * next_non_terminal * last_gae_lam_c
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            self.complex_advantages[step] = last_gae_lam_c
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        self.complex_returns = self.complex_advantages + self.complex_values

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        obs_no_norm: np.ndarray = None,
        next_obs_no_norm: np.ndarray = None,
        complexity: np.ndarray = None,
        complexity_value: th.Tensor = None,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        if obs_no_norm is not None:
            self.observations_no_norm[self.pos] = np.array(obs_no_norm).copy()
            self.next_observations_no_norm[self.pos] = np.array(next_obs_no_norm).copy()
        if complexity is not None:
            self.complexities[self.pos] = np.array(complexity).copy()
        if complexity_value is not None:
            self.complex_values[self.pos] = complexity_value.clone().cpu().numpy().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def add_complexities(self,complexities: th.Tensor,complex_values: Optional[th.Tensor] = None):
        self.complexities = complexities.clone().cpu().numpy()
        if complex_values is not None:
            self.complex_values = complex_values.clone().cpu().numpy().flatten()

    def compute_complexities(self, means: np.ndarray, bootstrapped: np.ndarray, true_obs = None):
        if true_obs is None:
            true_obs = self.next_observations_no_norm
            self.complexities = -np.abs(means - true_obs).sum(axis=-1)
        else:
            self.complexities = -np.abs(means-true_obs).mean(axis=-1)
        self.mean_complexity = self.complexities.mean()
        self.complexities = self.complexities.reshape(self.rewards.shape)
        self.complexities += bootstrapped
        self.complexities.sum(axis=-1)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "episode_starts",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "complexities",
                "complex_advantages",
                "complex_returns",
                "complex_values",
                "observations_no_norm",
                "next_observations_no_norm"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.dones[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.episode_starts[batch_inds],
            self.observations_no_norm[batch_inds],
            self.next_observations_no_norm[batch_inds],
            self.complexities[batch_inds].flatten(),
            self.complex_advantages[batch_inds].flatten(),
            self.complex_returns[batch_inds].flatten(),
            self.complex_values[batch_inds].flatten()
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

