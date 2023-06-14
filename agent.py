import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import printing
import strategy_pattern
import visualisation
from environment import PongEnvironment
from observer_pattern import Observable, Observer
from pong import Pong
from pong_observer import PongObserver
from tuples import Transition, PongObservationUpdate
from strategy_pattern import Strategy

env = PongEnvironment()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations=None, n_actions=None, should_print=True):
        super(DQN, self).__init__()

        if Strategy.NETWORK_STRUCTURE in strategy_pattern.strategies:
            layer_dims = strategy_pattern.strategies[Strategy.NETWORK_STRUCTURE]()
        else:
            hidden_layer_size = 1024
            hidden_layer_amount = 2
            hidden_layers = [hidden_layer_size for _ in range(hidden_layer_amount)]
            layer_dims = [n_observations] + hidden_layers + [n_actions]

        if should_print:
            printing.info(f"using network structure: {layer_dims}")

        self.layers = nn.ModuleList(
            [nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)]
        ).to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i].forward(x)
            x = F.relu(x)
        x = self.layers[-1].forward(x)
        return x


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
MULT = 5

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3
REPLAY_MEMORY_SIZE = 1000 * MULT

steps_done = 0


class DQNAgent(Observable, Observer):
    def __init__(self, n_observations=None, n_actions=None, eval_mode=False):
        super().__init__()

        n_obs, n_act = PongObserver.get__n_obs_n_act()
        if n_observations is None:
            printing.info("using inferred n_observations")
            n_observations = n_obs
        if n_actions is None:
            printing.info("using inferred n_actions")
            n_actions = n_act

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.eval_mode = eval_mode

        self.policy_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
        self.target_net = DQN(n_observations=n_observations, n_actions=n_actions, should_print=False).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.steps_done = 0
        self.episode = 0
        self.episode_durations = []
        self.should_train = 0

        self.rewards = []
        self.reward_this_episode = 0
        self.steps_this_episode = 0

        self.loss = []

        self.last_action_value = 0

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))

    def next_episode(self):
        self.episode += 1
        self.rewards.append(self.reward_this_episode)
        self.reward_this_episode = 0
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        printing.analysis(f"eps_threshold: {eps_threshold}")
        printing.analysis(f"buffer-len: {len(self.memory)}")
        printing.analysis(f"last_action_value: {self.last_action_value}")

    def select_action(self, state, convert_to_tensor=False):
        if convert_to_tensor:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)

        if sample > eps_threshold or self.eval_mode:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                q_values = self.policy_net(state)
                self.last_action_value = q_values.max(1)[0].item()
                action_id = q_values.max(1)[1].view(1, 1).item()
                self.notify_observers(action_id)
                return action_id
        else:
            action_id = random.randint(0, self.n_actions - 1)
            self.notify_observers(action_id)
            return action_id

    def train(self, transition: Transition):
        self.steps_done += 1
        self.reward_this_episode += transition.reward
        self.steps_this_episode += 1
        # print(f"memory len: {len(self.memory)}")

        state, action, next_state, reward = transition
        done = next_state is None
        if done:
            printing.debug("DONE")
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([action], device=device, dtype=torch.long)
        if not done:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            self.select_action(next_state)
            self.do_train()
        reward = torch.tensor([reward], device=device)

        self.memory.push(state, action, next_state, reward)

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

    def do_train(self):
        if self.eval_mode:
            return

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def update(self, update: PongObservationUpdate):
        transition, state = update
        if transition is not None:
            self.train(transition)
            state, action, next_state, reward = transition
        if state is not None:
            self.select_action(state, convert_to_tensor=True)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute MSE loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        lambda_reg = 0.01
        l2_reg = sum(torch.sum(param ** 2) for param in self.policy_net.parameters())

        # Uncomment if you want to use L2 regularization
        # loss += lambda_reg * l2_reg
        self.loss.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def plot_results(self):
        visualisation.plot_results(self.rewards, title='Smoothed Rewards per Episode', xlabel='Episodes',
                                   ylabel='Rewards')
        visualisation.plot_results(self.loss, title='Smoothed Loss per Step', xlabel='Steps', ylabel='Loss')


def normalize(batch):
    # Compute the mean and standard deviation for each feature
    mean = batch.mean(dim=0, keepdim=True)
    std = batch.std(dim=0, keepdim=True)

    # Normalize the data
    return (batch - mean) / (std + 1e-7)  # adding a small value to avoid division by zero

