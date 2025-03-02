"""Visualization Module.

Module providing visualization utilities for the project.
"""

import matplotlib
import matplotlib.pyplot as plt
import torch

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_durations(episode_durations: list[int], show_result: bool = False) -> None:
    """Plot the durations of episodes.

    Args:
        episode_durations (list[int]): List of episode durations.
        show_result (bool): Flag to indicate if the result should be shown.
            Defaults to False.

    Returns:
        None
    """
    minimum_for_average = 100
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= minimum_for_average:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_reward(average_rewards: list[tuple[int, float]], std_deviation_rewards: list[tuple[int, float]]) -> None:
    """Plot the average rewards with standard deviation.

    Args:
        average_rewards (list[tuple[int, float]]): List of average rewards.
        std_deviation_rewards (list[tuple[int, float]]): List of standard deviation of rewards.

    Returns:
        None
    """
    # Extract episodes, rewards, and standard deviations
    episodes = [ar.episode for ar in average_rewards]
    rewards = [ar.reward for ar in average_rewards]
    std_devs = [sd.reward for sd in std_deviation_rewards]

    # Plot the rewards over episodes with standard deviation
    plt.figure()
    plt.errorbar(episodes, rewards, yerr=std_devs, label="Average Reward", fmt='-o')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over Episodes")
    plt.legend()
    plt.show()
