"""Train the model.

This script provides training and testing functions for the model.


"""

from itertools import count

import gymnasium
import numpy
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from collections import namedtuple
from federated_learner import device
from federated_learner.agents.base_agent import DQNAgent


class LearningSuite:
    """Class for training and evaluating a DQN agent in a given environment.

    This class provides methods to train the agent, evaluate its performance,
    and fill the replay buffer with random actions. The training process
    involves interacting with the environment, storing transitions, and
    optimizing the model. The evaluation process measures the agent's
    performance over a specified number of episodes.
    """

    def __init__(
        self, cfg: DictConfig, agent: DQNAgent, env: gymnasium.Env, seed: int = 42
    ) -> None:
        """Initializes the Federated Learner.

        Args:
            agent (DQNAgent): The agent to be trained.
            env (gymnasium.Env): The environment in which the agent operates.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
        """
        self.__env = env
        self.__agent = agent
        self.__seed = cfg.experiment.seed
        self.__cfg = cfg

    @property
    def agent(self) -> DQNAgent:
        """Returns the agent instance.

        This method provides access to the private __agent attribute.

        Returns:
            Agent: The agent instance.
        """
        return self.__agent

    def train(self) -> int:
        """Trains the agent in the environment.

        Initializes the environment, performs actions, stores transitions,
        and optimizes the model until the episode is done.

        Returns:
            int: The number of steps taken in the episode.
        """
        # Initialize the environment and get its state
        state, _ = self.__env.reset(seed=self.__seed)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = self.__agent.select_action(state)
            observation, reward, terminated, truncated, _ = self.__env.step(
                action.item()
            )
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            self.__agent.remember(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.__agent.optimize_model()
            self.__agent.soft_update()

            if done:
                break
        return t + 1

    def test(self, episodes: int) -> tuple[numpy.float64, numpy.float64]:
        """Evaluates the performance of a DQN agent in a given environment.

        Args:
            episodes (int): The number of episodes to evaluate the agent.

        Returns:
            None
        """
        total_rewards = []
        # no gradients needed
        with torch.no_grad():
            for _ in range(episodes):
                state, _ = self.__env.reset(seed=self.__seed)

                state = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                total_reward = 0
                done = False
                while not done:
                    action = self.__agent.select_action(state)
                    observation, reward, terminated, truncated, _ = self.__env.step(
                        action.item()
                    )
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(
                            observation, dtype=torch.float32, device=device
                        ).unsqueeze(0)

                    state = next_state
                    total_reward += reward
                total_rewards.append(total_reward)
        return numpy.mean(total_rewards), numpy.std(total_rewards)

    def fill_buffer(self) -> None:
        """Fills the agent's replay buffer with random actions.

        Args:
            env (gymnasium.Env): The environment in which the agent will perform actions.
            agent (DQNAgent): The DQN agent whose replay buffer will be filled.
            seed (int): The seed to use for the environment. Defaults to 42.
        """
        # no gradients needed
        done = True
        with torch.no_grad():
            for _ in tqdm(range(self.__agent.memory.capacity), position=1, leave=False):
                if done:
                    state, info = self.__env.reset(seed=self.__seed)
                    state = torch.tensor(
                        state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    done = False
                action = torch.tensor(
                    [[numpy.random.choice(self.__agent.action_dim)]],
                    device=device,
                    dtype=torch.long,
                )
                observation, reward, terminated, truncated, _ = self.__env.step(
                    action.item()
                )
                reward = torch.tensor([reward], device=device, dtype=torch.float32)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                # Store the transition in memory
                self.__agent.remember(state, action, next_state, reward)

                # Move to the next state
                state = next_state

    def train_over_episodes(self, epochs: int | None = None) -> tuple[list, list, list]:
        """Trains the agent over a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the agent.
        """
        AverageReward = namedtuple("AverageReward", ("episode", "reward"))
        average_reward_list = []
        std_reward_list = []
        episode_durations = []
        if epochs is None:
            epochs = self.__cfg.experiment.num_episodes

        for i_episode in tqdm(range(epochs), position=0):
            # Initialize the environment and get its state
            if i_episode % self.__cfg.experiment.test_interval == 0:
                average_reward, std_reward = self.test(
                    self.__cfg.experiment.test_episodes
                )
                print(
                    f"Episode {i_episode} --> Average Total Reward (Evaluation):"
                    f" {average_reward} (±{std_reward})"
                )
                average_reward_list.append(AverageReward(i_episode, average_reward))
                std_reward_list.append(AverageReward(i_episode, std_reward))
            steps = self.train()
            episode_durations.append(steps)
        average_reward, std_reward = self.test(100)
        print(
            f"Episode {i_episode} --> Average Total Reward (Evaluation):"
            f" {average_reward} (±{std_reward})"
        )
        average_reward_list.append(AverageReward(i_episode, average_reward))
        std_reward_list.append(AverageReward(i_episode, average_reward))
        return average_reward_list, std_reward_list, episode_durations
