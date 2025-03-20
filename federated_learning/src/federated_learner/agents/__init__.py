"""Agent Module.

Module for different agent definitions and their configurations.
"""

from federated_learner.agents.dqn import DeepQNetwork

agent_registry = {"DeepQNetwork": DeepQNetwork}
