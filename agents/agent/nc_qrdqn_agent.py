import torch
from torch.optim import Adam

from agents.model import ncQRDQN
from agents.utils import disable_gradients, update_params,\
    calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class ncQRDQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=200, kappa=1.0, lr=5e-5, memory_size=10**6,
                 gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000,
                 epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0,exploration=False, star = False):
        super(ncQRDQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = ncQRDQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)
        # Target network.
        self.target_net = ncQRDQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

        # Fixed fractions.
        taus = torch.arange(
            0, N+1, device=self.device, dtype=torch.float32) / N
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, N)

        self.N = N
        self.kappa = kappa

    def learn(self):
        self.learning_steps += 1

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        quantile_loss, mean_q , q75, q25= self.calculate_loss(
            states, actions, rewards, next_states, dones)

        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)
            self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)
            self.writer.add_scalar("stats/75Q", q75, 4*self.steps)
            self.writer.add_scalar("stats/25Q", q25, 4*self.steps)

    def calculate_loss(self, states, actions, rewards, next_states, dones):

        # Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = evaluate_quantile_at_action(
            self.online_net(states=states),
            actions)
        assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_q = self.target_net.calculate_q(states=next_states)
            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net(states=next_states),
                next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (
                self.batch_size, 1, self.N)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (self.batch_size, 1, self.N)

        td_errors = target_sa_quantiles - current_sa_quantiles
        # print("target_q", target_sa_quantiles[0])
        # print("current_q", current_sa_quantiles[0].T)
        # print("td_errors", td_errors[0,:,0])
        
        assert td_errors.shape == (self.batch_size, self.N, self.N)
        # print("target_q",target_sa_quantiles[0])
        # print("current_q", current_sa_quantiles[0].T)


        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, self.tau_hats, self.kappa)
        # print("quantile_loss",quantile_huber_loss)
        return quantile_huber_loss, next_q.detach().mean().item(), torch.quantile(next_sa_quantiles, .75).detach().item(), torch.quantile(next_sa_quantiles, .25).detach().item()
