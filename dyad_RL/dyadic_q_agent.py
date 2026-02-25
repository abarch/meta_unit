import numpy as np


class DyadicQAgent:

    def __init__(self,
                 state_bins=10,
                 n_actions=6,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=0.1):

        self.state_bins = state_bins
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q = np.zeros(
            (state_bins,
             state_bins,
             state_bins,
             state_bins,
             n_actions)
        )

    def discretize(self, state):

        return tuple(
            int(x * (self.state_bins - 1))
            for x in state
        )

    def select_action(self, state):

        s = self.discretize(state)

        if np.random.rand() < self.epsilon:

            return np.random.randint(self.n_actions)

        return np.argmax(self.q[s])

    def update(self, state, action, reward, next_state):

        s = self.discretize(state)
        ns = self.discretize(next_state)

        best_next = np.max(self.q[ns])

        td_target = reward + self.gamma * best_next

        td_error = td_target - self.q[s][action]

        self.q[s][action] += self.alpha * td_error