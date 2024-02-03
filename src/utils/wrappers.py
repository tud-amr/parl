import gymnasium
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers.normalize import RunningMeanStd, update_mean_var_count_from_moments
from sb3_contrib.common.wrappers import TimeFeatureWrapper  # noqa: F401 (backward compatibility)
from src.envs.grid_envs import OBJECT_TO_IDX, COLOR_TO_IDX
#
# class DoneOnSuccessWrapper(gym.Wrapper):
#     """
#     Reset on success and offsets the reward.
#     Useful for GoalEnv.
#     """
#
#     def __init__(self, env: gym.Env, reward_offset: float = 0.0, n_successes: int = 1):
#         super().__init__(env)
#         self.reward_offset = reward_offset
#         self.n_successes = n_successes
#         self.current_successes = 0
#
#     def reset(self):
#         self.current_successes = 0
#         return self.env.reset()
#
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         if info.get("is_success", False):
#             self.current_successes += 1
#         else:
#             self.current_successes = 0
#         # number of successes in a row
#         done = done or self.current_successes >= self.n_successes
#         reward += self.reward_offset
#         return obs, reward, done, info
#
#     def compute_reward(self, achieved_goal, desired_goal, info):
#         reward = self.env.compute_reward(achieved_goal, desired_goal, info)
#         return reward + self.reward_offset
#


# Observation wrapper to convert a discrete observation space to a one-hot
class OneHotObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(OneHotObservation, self).__init__(env)
        self.num_obs = self.observation_space.n
        self.observation_space = spaces.Box(0, 1, (self.observation_space.n,), dtype=np.float32)

    def observation(self, observation):
        obs = np.zeros(self.num_obs)
        obs[observation] = 1
        return obs

class TransposedObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposedObservation, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, term, trunc, info

    def reset(self):
        return self.env.reset()

class RescaleWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.rescale = 10
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward*self.rescale, term,trunc, info

class MujocoWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return np.clip(obs,-10,10), np.clip(reward,-10,10), term,trunc, info

    def reset(self):
        return self.env.reset()


class LavaNegRewWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        if done and reward==0:
            reward = -0.5
        # elif not reward:
        #     reward = -0.05
        elif reward > 0:
            reward = 5
        return obs, reward, term,trunc, info


class LavaNegRewWrapper2(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if reward<0 and not term:
            reward = 0
        if reward > 0:
            reward = 1
        return obs, reward, term,trunc, info

class DynObsWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        if done and reward==0:
            reward = -1
        # elif not reward:
        #     reward = -0.05
        elif reward > -0.5:
            reward *= 5
        return obs, reward, term, trunc, info


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self.unnormalized_obs = None
        self.last_unnormalized_obs = None

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        self.last_unnormalized_obs = np.copy(self.unnormalized_obs)
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        self.unnormalized_obs = np.copy(obs)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def unnormalize(self, obs):
        return obs * np.sqrt(self.obs_rms.var + self.epsilon) + self.obs_rms.mean

class FullyObsReduceWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=env.observation_space.spaces["image"].high.max(),
            shape=(self.env.width, self.env.height, 2),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )
        full_grid = np.concatenate([np.expand_dims(full_grid[:, :, 0], -1), np.expand_dims(full_grid[:, :, 2], -1)], axis=2)
        return {**obs, "image": full_grid}

class NormalizeWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.initial_obs = env.reset()[0]
    def observation(self, obs):
        return obs - self.initial_obs

class FlattenObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)


class FlattenObservationReduce(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        """ The observation is of shape MxNx3, where M and N are the dimensions of the image and 3 is the number of
        channels. Before flattening, we remove the second channel"""
        observation = np.concatenate([np.expand_dims(observation[:, :, 0],-1), np.expand_dims(observation[:, :, 2],-1)], axis=2)
        return spaces.flatten(self.env.observation_space, observation)

class FlattenConcatenate(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that flattens the observation and concatenates it with 'alpha' observation."""

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        observation_space = env.observation_space.spaces['image']
        observation_space = spaces.flatten_space(observation_space)
        # The space in self.observation_space is a Box, and we want to define a new Box with one extra dimension
        # for the alpha observation. We do this by creating a new Box with the same bounds as the original Box,
        # but with one extra dimension.
        self.observation_space = spaces.Box(
            low=np.concatenate([observation_space.low, np.asarray([0])]),
            high=np.concatenate([observation_space.high, np.asarray([1])]),
            dtype=observation_space.dtype)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation concatenated with the input
        """
        img = observation['image']
        flatobs = spaces.flatten(self.observation_space, img)
        return np.concatenate([flatobs,observation['alpha']])

def simplifier_static(obs):
    # Take the grid image, remove the last channel, and change the switch id to 4 if state is 0 or 6 if state is 1
    switch_states = obs[:,17]
    if switch_states[0] == 1:
        obs[:,17] = 0
    # Remove all entries with odd indices
    obs = obs[:,::2]
    # Remove all entries equal to 2
    obs = obs[obs != 2]
    return obs

def simplifier_agent(obs):
    # Take the grid image, find the agent and keep only the agent position and direction
    agent_pos = np.where(obs == 10)
    agent_dir = obs[agent_pos[0], agent_pos[1] + 1]
    # The position is an index in a flattened array, so we need to convert it to a (x,y) coordinate pair
    agent_pos = np.unravel_index(agent_pos[1]//2, (9,7))
    obs = np.zeros((3,))
    # Construct a new obs vector of shape (N,2) where N is the number of observations in the original obs vector
    obs[0] = agent_pos[0]
    obs[1] = agent_pos[1]
    obs[2] = agent_dir

    # Assign the agent position and direction to the new obs vector in every row
    return obs