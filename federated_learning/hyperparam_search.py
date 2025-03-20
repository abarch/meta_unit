"""Train Agent with hyperparameter search.

We use Hydra for configuration management.
For configuring the hyperparameter search, we use Optuna.
"""

import hydra
from collections import namedtuple
import os
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from federated_learner.exp_manager import make_agent, make_env
from federated_learner.learning import LearningSuite

os.environ["HYDRA_FULL_ERROR"] = "1"


# Starting point for training agents.
# Decorator captures console arguments where config paths are provided.
# Based on hydra configuration manager. Creates an outputs directory for each run.
# Example command:
#       python train_agent.py
#       python train_agent.py -m
@hydra.main(version_base="1.2", config_path="./config", config_name="default")
def train(cfg: DictConfig):
    # root_path = os.getcwd()
    print(OmegaConf.to_yaml(cfg))

    env = make_env(cfg)
    agent = make_agent(cfg, env)

    SEED = 42
    learn: LearningSuite = LearningSuite(agent, env, SEED)
    num_episodes = 5
    AverageReward = namedtuple("AverageReward", ("episode", "reward"))
    average_rewards = []
    std_deviation_rewards = []
    episode_durations = []

    print("Filling the buffer with random actions")
    learn.fill_buffer()
    print("Buffer filled")
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
        if i_episode % 25 == 0:
            average_reward, std_deviation_reward = learn.test(100)
            print(
                f"Episode {i_episode} --> Average Total Reward (Evaluation): {average_reward}"
            )
            average_rewards.append(AverageReward(i_episode, average_reward))
            std_deviation_rewards.append(AverageReward(i_episode, average_reward))
        steps = learn.train()
        episode_durations.append(steps)
    average_reward, std_deviation_reward = learn.test(100)
    print(
        f"Episode {i_episode} --> Average Total Reward (Evaluation): {average_reward}"
    )
    average_rewards.append(AverageReward(i_episode, average_reward))
    std_deviation_rewards.append(AverageReward(i_episode, average_reward))

    print("Complete")
    # reward = model_learn(cfg, agent)
    reward = 0

    # Return is for optuna sweeper, in hydra decorator
    return reward


if __name__ == "__main__":
    train()
