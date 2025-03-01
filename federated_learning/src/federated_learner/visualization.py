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

plt.ion()


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
