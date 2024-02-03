#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import os
from torch.distributions import Categorical,Normal
DEVICE = torch.device('cpu') if not torch.cuda.is_available() else torch.device(0)

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(DEVICE)
    return x


def range_tensor(end):
    return torch.arange(end).long().to(DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param._grad = src_param.grad.clone()


# adapted from https://github.com/pytorch/pytorch/issues/12160
def batch_diagonal(input):
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    # stride and copy the input to the diagonal
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input):
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action):
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    def __init__(self, logits):
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=torch.Size([])):
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    def __init__(self, network=None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=DEVICE))

    def add(self, op):
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef):
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network):
        for grad, param in zip(self.grads, network.parameters()):
            param._grad = grad.clone()

    def zero(self):
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    def __init__(self, network=None, n=0, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, torch.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def add(self, op):
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, Grads):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add(op_grad)
        elif isinstance(op, torch.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def mean(self):
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad


def escape_float(x):
    return ('%s' % x).replace('.', '\.')


def get_state_kl_bound_sgld(network, batch_states, batch_action_means, eps, steps, stdev):
    #SGDL Solver for minimizing KL distance. Adapted from https://github.com/huanzhang12/SA_PPO/
    batch_action_means = batch_action_means.detach().to(network.device)
    # upper and lower bounds for clipping
    states_ub = batch_states + eps
    states_lb = batch_states - eps
    step_eps = eps / steps
    # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
    beta = 1e-5
    noise_factor = np.sqrt(2 * step_eps * beta)
    noise = torch.randn_like(batch_states.float()) * noise_factor
    var_states = (batch_states.clone() + noise.sign() * step_eps).detach().requires_grad_()
    if hasattr(network, 'get_distribution'):
        distribution = network.get_distribution(batch_states.to(network.device))
        if isinstance(distribution.distribution,Categorical):
            if batch_action_means.ndim<distribution.distribution.probs.ndim:
                acts = torch.gather(distribution.distribution.probs,dim=1,index=batch_action_means.unsqueeze(-1).long()).detach()
            else:
                acts = torch.gather(distribution.distribution.probs,dim=1,index=batch_action_means.long()).detach()
        else:
            acts = distribution.distribution.cdf(batch_action_means).detach()
        for i in range(steps):
            distribution = network.get_distribution(var_states.to(network.device))
            if isinstance(distribution.distribution, Categorical):
                if batch_action_means.ndim < distribution.distribution.probs.ndim:
                    acs = torch.gather(distribution.distribution.probs, dim=1,
                                        index=batch_action_means.unsqueeze(-1).long())
                else:
                    acs = torch.gather(distribution.distribution.probs, dim=1, index=batch_action_means.long())
            else:
                acs = distribution.distribution.cdf(batch_action_means)
            diff = acs.cpu() - acts.cpu()
            kl = (diff * diff).sum(axis=-1, keepdim=True).mean()
            # Need to clear gradients before the backward() for policy_loss
            kl.backward()
            # Reduce noise at every step.
            noise_factor = np.sqrt(2 * step_eps * beta) / (i+2)
            # Project noisy gradient to step boundary.
            update = (var_states.grad + noise_factor * torch.randn_like(var_states)).sign() * step_eps
            var_states.data += update
            # clip into the upper and lower bounds
            var_states = torch.max(var_states, states_lb)
            var_states = torch.min(var_states, states_ub)
            var_states = var_states.detach().requires_grad_()
        network.zero_grad()
        # diff = (net(var_states.requires_grad_(False))[0] - batch_action_means)
    else:
        for i in range(steps):
            # Find a nearby state new_phi that maximize the difference
            acs, _, _ = network.actor.get_action_dist_params(var_states.to(network.device))
            diff = acs.cpu() - batch_action_means.cpu()
            kl = (diff * diff).sum(axis=-1, keepdim=True).mean()
            # Need to clear gradients before the backward() for policy_loss
            kl.backward()
            # Reduce noise at every step.
            noise_factor = np.sqrt(2 * step_eps * beta) / (i + 2)
            # Project noisy gradient to step boundary.
            update = (var_states.grad + noise_factor * torch.randn_like(var_states)).sign() * step_eps
            var_states.data += update
            # clip into the upper and lower bounds
            var_states = torch.max(var_states, states_lb)
            var_states = torch.min(var_states, states_ub)
            var_states = var_states.detach().requires_grad_()
        network.zero_grad()
        # diff = (net(var_states.requires_grad_(False))[0] - batch_action_means)
    return var_states


def get_actor_parameters(agent):
    network = agent.network
    actor_body = network.actor_body.state_dict()
    fc_action = network.fc_action.state_dict()
    return {'body':actor_body,'fc':fc_action}

