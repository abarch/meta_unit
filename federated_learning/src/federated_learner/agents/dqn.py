"""Deep Q-Network implementation."""

import torch


class DeepQNetwork(torch.nn.Module):
    """Deep Q-Network implementation."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        """Initializes the DQN agent.

        Args:
            n_observations (int): Number of observations from the environment.
            n_actions (int): Number of possible actions the agent can take.
        """
        super(DeepQNetwork, self).__init__()
        self.layer_1 = torch.nn.Linear(n_observations, 64)
        self.layer_2 = torch.nn.Linear(64, 32)
        self.layer_3 = torch.nn.Linear(32, 32)  # Latent layer
        self.layer_4 = torch.nn.Linear(32, 64)
        self.layer_5 = torch.nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network layers.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
        """
        x = torch.nn.functional.gelu(self.layer_1(x))
        x = torch.nn.functional.gelu(self.layer_2(x))
        x = torch.nn.functional.gelu(self.layer_3(x))
        x = torch.nn.functional.gelu(self.layer_4(x))
        return self.layer_5(x)
