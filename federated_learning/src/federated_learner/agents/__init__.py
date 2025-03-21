"""Agent Module.

Module for different agent definitions and their configurations.
"""

from federated_learner.agents.base_agent import DQNAgent, AgentConfig
from federated_learner.agents.dqn import DeepQNetwork

agent_registry = {
    "DQNAgent": {
        "agent": DQNAgent,
        "model": {"DeepQNetwork": DeepQNetwork},
        "config": AgentConfig,
    }
}
