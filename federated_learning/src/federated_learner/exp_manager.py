"""Experiment manager module."""

from federated_learner.agents.base_agent import DQNAgent, AgentConfig
from federated_learner.agents import agent_registry
from omegaconf import DictConfig
import gymnasium


def make_env(cfg: DictConfig) -> gymnasium.Env:
    """Creates and returns a Gymnasium environment.

    Args:
        cfg (DictConfig): Configuration dictionary (currently unused).

    Returns:
        gymnasium.Env: An instance of the Acrobot-v1 environment.
    """
    return gymnasium.make(cfg.environment.env_id)


def make_agent(cfg: DictConfig, env: gymnasium.Env) -> DQNAgent:
    """Creates and initializes a DQNAgent with the given configuration.

    Args:
        cfg (DictConfig): Configuration object containing agent parameters.
        env (gymnasium.Env): The environment in which the agent will operate.

    Returns:
        DQNAgent: An instance of the DQNAgent initialized with the specified
            configuration and environment.

    Raises:
        KeyError: If the specified agent model is not found in the registry.
    """
    state, _ = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n
    config = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_end=cfg.agent.epsilon_end,
        epsilon_decay=cfg.agent.epsilon_decay,
        buffer_size=cfg.agent.buffer_size,
        batch_size=cfg.agent.batch_size,
    )

    return DQNAgent(config, agent_registry[cfg.agent.model])
