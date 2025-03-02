"""Utility functions for training and testing agents."""

import gymnasium
from tqdm import tqdm
import torch
import numpy

from federated_learner.agent import DQNAgent
from federated_learner import device


def test_agent(env: gymnasium.Env, agent: DQNAgent, seed: int = 42) -> None:
    """Evaluates the performance of a DQN agent in a given environment.

    Args:
        env (gymnasium.Env): The environment in which the agent will be tested.
        agent (DQNAgent): The DQN agent to be evaluated.
        seed (int): The seed to use for the environment. Defaults to 42.

    Returns:
        None
    """
    total_rewards = []
    num_episodes_eval = 100
    # no gradients needed
    with torch.no_grad():
        for _ in range(num_episodes_eval):
            state, _ = env.reset(seed=seed)

            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward = 0
            done = False
            while not done:
                action = agent.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
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

def fill_buffer(env: gymnasium.Env, agent: DQNAgent, seed: int = 42) -> None:
    """Fills the agent's replay buffer with random actions.

    Args:
        env (gymnasium.Env): The environment in which the agent will perform actions.
        agent (DQNAgent): The DQN agent whose replay buffer will be filled.
        seed (int): The seed to use for the environment. Defaults to 42.
    """
    # no gradients needed
    done = True
    with torch.no_grad():
        for _ in tqdm(range(agent.memory.capacity)):
            if done:
                state, info = env.reset(seed=seed)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                done = False
            action =  torch.tensor(
                [[numpy.random.choice(agent.action_dim)]],
                device=device,
                dtype=torch.long,
            )
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            agent.remember(state, action, next_state, reward)

            # Move to the next state
            state = next_state
