from collections import deque
import numpy as np
import torch


def update_params(optim, loss, networks, retain_graph=False,
                  grad_cliping=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        for net in networks:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))

def calculate_quantile_loss(td_errors, taus):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate quantile loss element-wisely.
    element_wise_quantile_huber_loss = (
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * td_errors
    
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    return element_wise_quantile_huber_loss.sum(dim=1).mean()

def calculate_quantile_huber_loss(td_errors, taus, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    return element_wise_quantile_huber_loss.sum(dim=1).mean()


def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None].expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles

def evaluate_kheaddqn_at_action(pred, actions):
  # (K,batch_size,action_num)
    batch_size = pred.shape[1]
    K = pred.shape[0]
#     print('actions size is')
#     print(actions.shape)
#     print('pred size is ')
#     print(pred.shape)
#     print('\n')
    # Expand actions into (K, batch_size, 1).
    action_index = actions[None, ...].reshape(1,batch_size,1)
    action_index = action_index.expand(K, batch_size, 1)

    # Calculate quantile values at specified actions.
    current_sa_dqn = pred.gather(dim=2, index=action_index)


    assert current_sa_dqn.shape == (K, batch_size, 1)
    return current_sa_dqn

def evaluate_kheaddqn_at_action_(pred, actions):
  # (K,batch_size,action_num)
    batch_size = pred.shape[1]
    K = pred.shape[0]
#     print('actions size is')
#     print(actions.shape)
#     print('pred size is ')
#     print(pred.shape)
#     print('\n')
    # Expand actions into (K, batch_size, 1).

    # Calculate quantile values at specified actions.
    current_sa_dqn = pred.gather(dim=2, index=actions)


    assert current_sa_dqn.shape == (K, batch_size, 1)
    return current_sa_dqn

class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class LinearAnneaer:

    def __init__(self, start_value, end_value, num_steps):
        assert num_steps > 0 and isinstance(num_steps, int)

        self.steps = 0
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

        self.a = (self.end_value - self.start_value) / self.num_steps
        self.b = self.start_value

    def step(self):
        self.steps = min(self.num_steps, self.steps + 1)

    def get(self):
        assert 0 < self.steps <= self.num_steps
        return self.a * self.steps + self.b
