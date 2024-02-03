
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gymnasium as gym

import numpy as np
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.common.type_aliases import GymEnv, GymObs, GymStepReturn, MaybeCallback, Schedule, TensorDict
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions.normal import Normal
from torch import sigmoid
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor,FlattenExtractor, NatureCNN
from stable_baselines3.common.running_mean_std import RunningMeanStd
from src.utils.torch_networks import layer_init, MiniGridCNN
from src.utils.wrappers import simplifier_static, simplifier_agent


class DynamicModel(nn.Module):
    def __init__(
        self,
        environment: GymEnv,
        activation_fn: Type[nn.Module] = nn.ReLU(),
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        feature_extraction: Union[str,BaseFeaturesExtractor] = FlattenExtractor,
        batchnorm: bool = False,
        lr: Union[float, str] = 0.001,
        device: str = "cpu",
        dropout: float = 0.0,
        model_arch: List[int] = [128, 64, 128],
        simplify_obs: str = "None",
        entropy_estimation: str = "sum",
        epsilon: float = 0.0001,
    ) -> None:
        super().__init__()
        self.conv = False
        self.softplus = nn.Softplus()
        self.activation_fn = activation_fn
        self.sp = nn.Sigmoid()
        self.observation_space = environment.observation_space
        self.latent_dim = model_arch[int(len(model_arch)/2)]
        self.batchnorm = batchnorm
        self.std_scale = 2
        net_arch = model_arch
        self.out_layer_mean = None
        self.out_layer_std = None
        self.simplify_obs = simplify_obs
        self.model = None
        self.entropy_estimation = entropy_estimation
        _ = environment.reset()
        self.dropout = dropout
        try:
            self.initial_state = environment.get_original_obs()
        except:
            self.initial_state = _
        try:
            self.K = environment.observation_space.shape[-1]
        except:
            self.K = 1
        self._generate_architecture(environment.observation_space, environment.action_space, feature_extraction,
                                    net_arch)
        if isinstance(lr, str):
            self.learning_rate = float(lr.split("_")[1])
            self.decay_rate = True
        else:
            self.learning_rate = lr
            self.decay_rate = False
        self.optimizer = optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=self.learning_rate
        )
        self.device = th.device(device)
        self.to(self.device)
        self.epsilon = epsilon
        self.c = 0.5*(1 + th.log(2 * th.tensor(np.pi))).to(self.device)
        self.max_std = 1.0
        self.last_phi = None
        self.normalizer = RunningMeanStd(shape=(1,))
        self.has_encoder = False

    def _generate_architecture(self, observation_space, action_space, feature_extraction, net_arch):
        if action_space.__class__.__name__ == "Discrete":
            a_num = 1
        else:
            a_num = action_space.shape[0]
        if feature_extraction == FlattenExtractor:
            if observation_space.__class__.__name__ == "Discrete":
                # One hot encoding of discrete observations
                input_dims = observation_space.n
            else:
                input_dims = spaces.utils.flatdim(observation_space)
            layers = [layer_init(nn.Linear(input_dims+a_num, net_arch[0])), self.activation_fn]
            for i in range(int(len(net_arch)/2)):
                layers.append(layer_init(nn.Linear(net_arch[i], net_arch[i+1])))
                if self.batchnorm:
                    layers.append(nn.BatchNorm1d(net_arch[i + 1]))
                layers.append(self.activation_fn)
                if self.dropout>0:
                    layers.append(nn.Dropout(self.dropout))
            self.features_extractor = nn.Sequential(*layers)
            self.action_encoder = layer_init(nn.Linear(a_num, self.latent_dim))
            layers = []
            # net_arch[int(len(net_arch)/2)] *= 2
            for li,l in enumerate(net_arch[int(len(net_arch)/2)+1:]):
                layers.append(layer_init(nn.Linear(net_arch[int(len(net_arch)/2)+li],
                                                   net_arch[int(len(net_arch)/2)+li+1])))
                if self.batchnorm:
                    layers.append(nn.BatchNorm1d(net_arch[int(len(net_arch) / 2) + li + 1]))
                layers.append(self.activation_fn)
                if self.dropout>0:
                    layers.append(nn.Dropout(self.dropout))
            layers.append(layer_init(nn.Linear(net_arch[-1], input_dims)))
            if observation_space.__class__.__name__ == "Discrete":
                layers.append(self.sp)
            # self.fc_reshape = nn.Identity(layers[0].input_features, layers[0].input_features)
            self.mean_decoder = nn.Sequential(*layers)
            layers = []
            for li, l in enumerate(net_arch[int(len(net_arch) / 2) + 1:]):
                layers.append(layer_init(nn.Linear(net_arch[int(len(net_arch) / 2) + li],
                                                   net_arch[int(len(net_arch) / 2) + li + 1])))
                if self.batchnorm:
                    layers.append(nn.BatchNorm1d(net_arch[int(len(net_arch) / 2) + li + 1]))
                layers.append(self.activation_fn)
                if self.dropout>0:
                    layers.append(nn.Dropout(self.dropout))
            layers.append(layer_init(nn.Linear(net_arch[-1], input_dims)))
            layers.append(nn.ReLU())
            self.std_decoder = nn.Sequential(*layers)
        elif feature_extraction == MiniGridCNN:
            raise NotImplementedError


    def predict_mean_std(self,obs,action):
        if self.conv:
            raise NotImplementedError
        else:
            # Unsqueeze obs if needed
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            obs = th.cat([obs, action], dim=-1)
            phi = self.features_extractor(obs)
            # phi_a = th.tanh(self.action_encoder(action))
            # phi = th.cat([phi,phi_a], dim=1)
            # phi_r = self.activation_fn(self.rew_decoder(phi))
            # rew_std = self.sp(self.rew_std(phi_r))
            # rew_mean = self.rew_decoder(phi)
            # done = self.done_decoder(phi)
            mean = self.mean_decoder(phi)
            std = self.std_decoder(phi)+self.epsilon
        return mean, std #rew_mean, done

    def predict_mean(self,obs,action):
        if self.conv:
            raise NotImplementedError
        else:
            # Unsqueeze obs if needed
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            obs = th.cat([obs, action], dim=-1)
            phi = self.features_extractor(obs)
            self.last_phi = phi.detach()
            # rew_mean = self.rew_decoder(phi)
            # rew_mean = None
            # done = self.done_decoder(phi)
            mean = self.mean_decoder(phi)
        return mean#, rew_mean, done

    def predict_done_reward(self,obs=None,action=None):
        if self.conv:
            raise NotImplementedError
        else:
            # Unsqueeze obs if needed
            if obs is None:
                rew_mean = self.rew_decoder(self.last_phi)
                done = self.done_decoder(self.last_phi)
                return rew_mean, done
            else:
                with th.no_grad():
                    if len(obs.shape) == 1:
                        obs = obs.unsqueeze(0)
                    obs = th.cat([obs, action], dim=1)
                    phi = self.features_extractor(obs)
                rew_mean = self.rew_decoder(phi)
                done = self.done_decoder(phi)
            return rew_mean, done

    def build_distribution(self, obs, action):
        mean, std, _, _ = self.predict_mean_std(obs, action)
        return Normal(mean, std)

    def compute_std_frozen(self, obs=None, action=None):
        if self.conv:
            raise NotImplementedError
        else:
            with th.no_grad():
                if len(obs.shape) == 1:
                    obs = obs.unsqueeze(0)
                obs = th.cat([obs, action], dim=-1)
                phi = self.features_extractor(obs)
            std = self.std_decoder(phi)
        # distribution = Normal(mean, std+self.epsilon)
        std = std +self.epsilon
        return std

    def log_probs(self, obs, action, next_obs):
        # mean, std, rew, done = self.predict_mean_std(obs, action)
        mean, std = self.predict_mean_std(obs, action)
        distribution = Normal(mean, std)
        return distribution.log_prob(next_obs) #, rew, done

    def forward(self, obs: th.Tensor, action: np.ndarray) -> th.Tensor:
        # mean, std, rew_mean, done = self.predict_mean_std(obs, action)
        mean, std = self.predict_mean_std(obs, action)
        distribution = Normal(mean, std)
        prediction = distribution.sample()
        return prediction #, rew_mean, done

    def predict(self, obs: th.Tensor,action: th.Tensor) -> th.Tensor:
        with th.no_grad():
            # mean, std, reward, done = self.predict_mean_std(obs, action)
            mean, std = self.predict_mean_std(obs, action)
            prediction = Normal(mean, std).sample()
        return prediction #, reward, done

    def predict_complexity(self,obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        with th.no_grad():
            mean, std = self.predict_mean_std(obs, action)
            # std = th.div(std, self.mean+1)
            if self.entropy_estimation=="sum":
                std = th.sum(std, dim=-1) #th.max(std, dim=1)[0]*0.5 + th.mean(std,dim=1)*0.5 # + 1
                # complexities = std-1 #nn.functional.tanh(std)-1
                complexities = th.log(std) - np.log(self.epsilon*obs.shape[-1]) #2*nn.functional.tanh(std)
            elif self.entropy_estimation=="max":
                std = th.max(std, dim=-1)[0]*0.5 + th.mean(std,dim=-1)*0.5 + 1
                complexities = 2*th.log(std)
            #set to 0 if logstd is smaller than -5
            # logstd = th.where(logstd<-5, th.zeros_like(logstd), logstd)
            # complexities = logstd
            # complexities = th.sum(complexities, dim=1)+obs.shape[1]*self.c
            # get the maximum complexity of the batch
            # complexities = complexities#-0.5*th.log(th.as_tensor(1))
            # self.normalizer.update(complexities.cpu().numpy())
            # complexities = th.clip(complexities / np.sqrt(self.normalizer.var + self.epsilon), -1, 1).float()
        return complexities

class DynamicModelEncoder(nn.Module):
    def __init__(
            self,
            environment: GymEnv,
            activation_fn: Type[nn.Module] = nn.SiLU(),
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            feature_extraction: Union[str, BaseFeaturesExtractor] = FlattenExtractor,
            batchnorm: bool = False,
            lr: Union[float, str] = 0.001,
            device: str = "cpu",
            dropout: float = 0.2,
            model_arch: dict = {"encoder":[128, 16, 128], "model":[16, 16, 16]},
            simplify_obs: str = "None",
            entropy_estimation: str = "sum",
            epsilon: float = 0.0001,
            type: str = "autoencoder"
    ) -> None:
        super().__init__()
        self.conv = False
        self.softplus = nn.Softplus()
        self.activation_fn = activation_fn
        self.sp = nn.Sigmoid()
        self.observation_space = environment.observation_space
        self.latent_dim = model_arch["encoder"][int(len(model_arch["encoder"])/2)]
        self.batchnorm = batchnorm
        self.std_scale = 2
        net_arch = model_arch
        self.out_layer_mean = None
        self.out_layer_std = None
        self.simplify_obs = simplify_obs
        self.model = None
        self.entropy_estimation = entropy_estimation
        _ = environment.reset()
        self.dropout = dropout
        try:
            self.initial_state = environment.get_original_obs()
        except:
            self.initial_state = _
        self.latent_fn = nn.Tanh()

        self._generate_architecture(environment.observation_space, environment.action_space, feature_extraction,
                                    net_arch)
        if isinstance(lr, str):
            self.learning_rate = float(lr.split("_")[1])
            self.decay_rate = True
        else:
            self.learning_rate = lr
            self.decay_rate = False
        self.optimizer = optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=self.learning_rate
        )
        self.device = th.device(device)
        self.to(self.device)
        self.epsilon = epsilon
        self.has_encoder = True


    def _generate_architecture(self, observation_space, action_space, feature_extraction, net_arch):
        if action_space.__class__.__name__ == "Discrete":
            a_num = 1
        else:
            a_num = action_space.n
        if feature_extraction == FlattenExtractor:
            if observation_space.__class__.__name__ == "Discrete":
                # One hot encoding of discrete observations
                input_dims = observation_space.n
            else:
                input_dims = spaces.utils.flatdim(observation_space)
            # Initialise autoencoder
            layers = [layer_init(nn.Linear(input_dims, net_arch["encoder"][0])), self.activation_fn]
            if self.dropout>0:
                layers.append(nn.Dropout(self.dropout))
            for i in range(int(len(net_arch["encoder"])-1)):
                layers.append(layer_init(nn.Linear(net_arch["encoder"][i], net_arch["encoder"][i+1])))
                if self.batchnorm:
                    layers.append(nn.BatchNorm1d(net_arch["encoder"][i + 1]))
                if i==len(net_arch["encoder"])//2-1:
                    layers.append(self.latent_fn)
                else:
                    layers.append(self.activation_fn)
                    if self.dropout>0:
                        layers.append(nn.Dropout(self.dropout))
            layers.append(layer_init(nn.Linear(net_arch["encoder"][-1], input_dims)))
            self.encoder = nn.Sequential(*layers[0:len(layers)//2+1])
            self.decoder = nn.Sequential(*layers[len(layers)//2+1:])

            # Initialise model
            layers = [layer_init(nn.Linear(self.latent_dim+a_num, net_arch["model"][0])), self.activation_fn]
            for i in range(int(len(net_arch["model"])-2)):
                layers.append(layer_init(nn.Linear(net_arch["model"][i], net_arch["model"][i+1])))
                # if self.batchnorm:
                #     layers.append(nn.BatchNorm1d(net_arch[i + 1]))
                layers.append(self.activation_fn)
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            self.model_common = nn.Sequential(*layers)
            # net_arch[int(len(net_arch)/2)] *= 2
            self.mean_output = nn.Sequential(*[layer_init(nn.Linear(net_arch["model"][-2], net_arch["model"][-1])),
                                               nn.ReLU(),layer_init(nn.Linear(net_arch["model"][-1], self.latent_dim)),
                                               self.latent_fn])
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(observation_space.shape[0], 16, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (32, 4, 4)),
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0,
                                   output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0,
                                   output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, observation_space.shape[0], kernel_size=8, stride=4, padding=0,
                                   output_padding=0),
                nn.ReLU(),
            )
            latent_dim = 512
            layers = [layer_init(nn.Linear(latent_dim+a_num, net_arch["model"][0])), self.activation_fn]
            for i in range(int(len(net_arch["model"])-2)):
                layers.append(layer_init(nn.Linear(net_arch["model"][i], net_arch["model"][i+1])))
                layers.append(self.activation_fn)
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            self.model_common = nn.Sequential(*layers)
            self.mean_output = nn.Sequential(*[layer_init(nn.Linear(net_arch["model"][-2], net_arch["model"][-1])),
                                               nn.ReLU(),layer_init(nn.Linear(net_arch["model"][-1], latent_dim)),
                                               self.latent_fn])

    def predict_mean(self,obs,action):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        latent_obs = self.encoder(obs)
        latent_obs = th.cat([latent_obs, action], dim=-1)
        phi = self.model_common(latent_obs)
        mean = self.mean_output(phi)
        return mean

    def autoencoder(self, obs):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        latent_obs = self.encoder(obs)
        obs = self.decoder(latent_obs)
        return obs

    def forward(self,obs,action,next_obs):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        latent_obs = self.encoder(obs)
        reconstructed_obs = self.decoder(latent_obs)
        latent_obs = th.cat([latent_obs, action], dim=-1)
        phi = self.model_common(latent_obs)
        mean = self.mean_output(phi)
        with th.no_grad():
            y_hat = self.encoder(next_obs)
        return reconstructed_obs, y_hat, mean





class DynamicModelEnv(gym.Env):
    def __init__(self, env: GymEnv, model: DynamicModel, vectorized:bool = False,horizon: int = 10):
        self.env = env
        self.model = model
        self.horizon = horizon
        self.num_envs = 1
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        if env.observation_space.dtype == np.uint8:
            self.round = True
        else:
            self.round = False
        self.state = self.env.reset()
        if vectorized:
            self.vectorized = True
        else:
            self.vectorized = False

    def update_model(self, model: DynamicModel):
        self.model = model

    def reset(self,state=None,is_vec=False) -> (GymObs,dict):
        if state is not None:
            is_vec = is_vectorized_observation(state, self.observation_space)
            self.state = state
        else:
            self.state = th.as_tensor(self.env.reset()).to(self.model.device)
        self.state = th.as_tensor(self.state).to(self.model.device)
        if self.vectorized and not is_vec:
            self.state = self.state.unsqueeze(0)
        return self.state.cpu().numpy(),{}

    def step(self, action: np.ndarray):
        next_state, reward, done = self.model.predict(self.state,th.as_tensor(action).float().unsqueeze(dim=0).to(self.model.device))
        if self.round:
            next_state = th.round(next_state)
        self.state = next_state
        if self.vectorized:
            next_state = next_state.cpu().numpy()
            reward = reward.cpu().numpy()
            done = done.cpu().numpy()
            infos = [{'TimeLimit.truncated': False}]
            done = done > 0.5
            if done:
                state = self.env.reset()
                self.state = th.as_tensor(state).float().to(self.model.device)
            done = np.asarray([done])
        else:
            next_state = np.squeeze(next_state.cpu().numpy())
            reward = np.squeeze(reward.cpu().numpy())
            done = np.squeeze(done.cpu().numpy())
            infos = {'TimeLimit.truncated': False}
            done = done > 0.5
            if done:
                state = self.env.reset()
                self.state = th.as_tensor(state).float().squeeze().to(self.model.device)
        return next_state, reward, done, infos



