import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from agents.memory import LazyMultiStepMemory
from agents.utils import RunningMeanStats, LinearAnneaer
import pickle


class BaseAgent:

    def __init__(self, env, test_env, log_dir, num_steps=5 * (10 ** 7),
                 batch_size=32, memory_size=10 ** 6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0, exploration=False, tensorboard=False):

        self.env = env
        self.test_env = test_env


        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2 ** 31 - 1 - seed)

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        self.memory = LazyMultiStepMemory(
            memory_size, self.env.observation_space.shape,
            self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.r_path = os.path.join(self.summary_dir, 'return.pkl')
        self.r = []
        self.eval_r = []

        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping
        self.exploration = exploration

    def run(self):
        start_time = time.time()
        while True:
            self.train_episode(start_time)
            pkl_file = open(self.r_path, 'wb')
            pickle.dump([self.r,self.eval_r], pkl_file)
            pkl_file.close()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0 \
               and self.steps >= self.start_steps

    def is_greedy(self, eval=False):
        if eval:
            return np.random.rand() < self.epsilon_eval
        else:
            return self.steps < self.start_steps \
                   or np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.online_net.calculate_q(states=state).argmax().item()
        return action

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))

    def load_models(self, save_dir):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))

    def train_episode(self, start_time):

        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()
        while (not done) and episode_steps <= self.max_episode_steps:

            if self.steps % self.update_interval == 0:
                # Reset the noise of noisy net (online net only).
                self.online_net.reset_noise()
                
            if self.exploration:
                action = self.choose_action(state)

            elif self.is_greedy(eval=False):
                action = self.explore()

            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)
            
            self.memory.append(
                state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            current_time = time.time()

            p_hours, remainder = divmod(current_time - start_time, 3600)
            p_minutes, _ = divmod(remainder, 60)

            t_hours, remainder = divmod(int((current_time - start_time) * (self.num_steps - self.steps) / self.steps), 3600)
            t_minutes, _ = divmod(remainder, 60)

            if self.tensorboard:
                self.writer.add_scalar('return/train', self.train_return.get(), 4 * self.steps)
            self.r.append(self.train_return.get())
            print('Mean_return:{} Frame:{:.2f} M  Time_used {} Time_togo {}'.format(self.train_return.get(), self.steps*4/1000000, "{:.0f}H-{:.0f}M".format(p_hours,p_minutes), "{:.0f}H-{:.0f}M".format(t_hours,t_minutes)))


    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        if self.steps % self.eval_interval == 0:
            self.online_net.eval()
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))
            self.online_net.train()

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        # We log evaluation results along with training frames = 4 * steps.
        if self.tensorboard:
            self.writer.add_scalar('return/test', mean_return, 4 * self.steps)
        self.eval_r.append(mean_return)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def load_checkpoint(self):
        print("loading breakpoint")

        print(os.path.join(self.model_dir, "final"))
        self.load_models(os.path.join(self.model_dir, "final"))

        print(os.path.join(self.summary_dir,"return.pkl"))
        pkl_file = open(os.path.join(self.summary_dir,"return.pkl"), 'rb')
        summary = pickle.load(pkl_file)
        self.r = summary[0]
        self.eval_r = summary[1]
        print(self.r, self.eval_r)
        self.episodes = len(self.r) * 100
        self.steps = len(self.eval_r) * 250000
        self.epsilon_train.steps = min(self.epsilon_train.num_steps, self.steps+1)
        print(f"model in {self.steps*4//1000000}M steps, {self.episodes} episodes")
        pkl_file.close()

            
    def __del__(self):
        self.env.close()
        self.test_env.close()
        if self.tensorboard:
            self.writer.close()
