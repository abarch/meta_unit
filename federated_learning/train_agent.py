"""Train Agent with hyperparameter search.

We use Hydra for configuration management.
For configuring the hyperparameter search, we use Optuna.
"""

import hydra
import os
from omegaconf import DictConfig, OmegaConf
from federated_learner.exp_manager import make_agent, make_env
from federated_learner.learning import LearningSuite

os.environ["HYDRA_FULL_ERROR"] = "1"


# Starting point for training agents.
# Decorator captures console arguments where config paths are provided.
# Example command:
#       python train_agent.py
#       python train_agent.py -m
@hydra.main(version_base="1.3", config_path="./config", config_name="default")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    env = make_env(cfg)
    agent = make_agent(cfg, env)

    learn: LearningSuite = LearningSuite(cfg, agent, env)
    learn.train_over_episodes()
    print("Complete")


if __name__ == "__main__":
    train()
