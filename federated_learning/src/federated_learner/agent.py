"""Deep Q-Learning agent for interacting with the environment.

This module contains the implementation of the DQNAgent and all
other neccessary classes and functions for the agent to interact
with the environment.

"""

import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import device

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


@dataclass
class AgentConfig:
    """Configuration parameters for the Agent.

    Attributes:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        learning_rate (float): Learning rate for the agent.
        tau (float): Soft update parameter for the target network.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_final (float): Final exploration rate.
        epsilon_decay (int): Decay rate for exploration (steps).
        buffer_size (int): Size of the replay buffer.
        batch_size (int): Size of the batch for optimization.
    """

    state_dim: int
    action_dim: int
    learning_rate: float
    tau: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: int
    buffer_size: int
    batch_size: int


class ReplayMemory(object):
    """A class used to store and manage replay memory for an agent.

    Attributes:
        memory (deque): A deque to store transitions with a fixed capacity.
    """

    def __init__(self, capacity: int) -> None:
        """Initializes the agent with a memory capacity.

        Args:
            capacity (int): The maximum size of the memory deque.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: np.ndarray) -> None:
        """Push a new transition into the memory.

        Args:
            *args (np.ndarray): The transition data to be stored.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a batch of transitions from memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list[Transition]: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the number of elements in the memory.

        Returns:
            int: The number of elements in the memory.
        """
        return len(self.memory)


class DeepQNetwork(nn.Module):
    """Deep Q-Network implementation."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        """Initializes the DQN agent.

        Args:
            n_observations (int): Number of observations from the environment.
            n_actions (int): Number of possible actions the agent can take.
        """
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network layers.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
        """
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """Deep Q-Learning agent interacting with environment."""

    def __init__(self, config: AgentConfig, model: nn.Module) -> None:
        """Initialization."""
        self.action_dim = config.action_dim
        self.lr = config.learning_rate
        self.tau = config.tau
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.memory = ReplayMemory(config.buffer_size)
        self.model = model(config.state_dim, config.action_dim)
        self.policy_net = model(config.state_dim, config.action_dim).to(device)
        self.target_net = model(config.state_dim, config.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=config.learning_rate, amsgrad=True
        )
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """Selects an action based on the given state.

        Epislon-greedy policy is used to select the action.
        The epsilon dacay is implemented as follows:


        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)

        See:
            https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The action to be taken.
        """
        # Implementation of epsilon_decay

        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1

        if np.random.rand() > epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[np.random.choice(self.action_dim)]],
                device=device,
                dtype=torch.long,
            )

    def optimize_model(self) -> None:
        """Optimize the model.

        by sampling a batch from memory and performing
        a single step of the optimization. This includes computing the loss
        and performing backpropagation.
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self) -> None:
        """Soft update of the target network's weights.

        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        """Stores the experience in the replay buffer.

        Args:
            state (np.ndarray): The current state of the environment.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state of the environment.
        """
        self.memory.push(state, action, reward, next_state)
