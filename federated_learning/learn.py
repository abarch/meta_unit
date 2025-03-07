"""Learning script."""

from collections import namedtuple
from itertools import count

import gymnasium
import matplotlib.pyplot as plt
import torch

from src.federated_learner import device
from src.federated_learner.agent import AgentConfig, DeepQNetwork, DQNAgent
from src.federated_learner.utils import test_agent
from src.federated_learner.visualization import plot_average_rewards, plot_durations

episode_durations = []
num_episodes = 50

env = gymnasium.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent_config = AgentConfig(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-4,
    gamma=0.99,
    tau=0.005,
    epsilon_start=0.9,
    epsilon_decay=1000,
    epsilon_end=0.05,
    buffer_size=10000,
    batch_size=128,
)


agent = DQNAgent(agent_config, DeepQNetwork)

AverageReward = namedtuple("AverageReward", ("episode", "reward"))
average_rewards = []

plt.ion()

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    if i_episode % 100 == 0:
        average_reward = test_agent(env, agent)
        print(
            f"Episode {i_episode} --> Average Total Reward (Evaluation): {average_reward}"
        )
        average_rewards.append(AverageReward(i_episode, average_reward))

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
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

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()
        agent.soft_update()

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break


print("Complete")
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()

plot_average_rewards(average_rewards)
