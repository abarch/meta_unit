import numpy as np


class DyadicPianoEnv:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        # persistent talent differences
        self.talent_pitch = [
            self.rng.uniform(0.5, 1.0),
            self.rng.uniform(0.5, 1.0)
        ]

        self.talent_timing = [
            self.rng.uniform(0.5, 1.0),
            self.rng.uniform(0.5, 1.0)
        ]

        self.reset()

    def reset(self):

        self.state = np.array([
            self.rng.uniform(0.1, 0.3),  # pitch A1
            self.rng.uniform(0.1, 0.3),  # pitch A2
            self.rng.uniform(0.1, 0.3),  # timing A1
            self.rng.uniform(0.1, 0.3),  # timing A2
        ], dtype=np.float32)

        self.roles = [0, 1]  # 0=leader, 1=follower

        return self.state.copy()

    def step(self, action_A1, action_A2):

        prev_performance = self.performance()

        self.apply_action(0, action_A1)
        self.apply_action(1, action_A2)

        self.state = np.clip(self.state, 0, 1)

        new_performance = self.performance()

        reward_A1 = new_performance - prev_performance
        reward_A2 = new_performance - prev_performance

        return self.state.copy(), reward_A1, reward_A2

    def performance(self):

        pitch_mean = (self.state[0] + self.state[1]) / 2
        timing_mean = (self.state[2] + self.state[3]) / 2

        return 0.5 * pitch_mean + 0.5 * timing_mean

    def apply_action(self, agent, action):

        other = 1 - agent

        pitch_idx = agent
        timing_idx = agent + 2

        other_pitch = other
        other_timing = other + 2

        if action == 0:  # übe timing

            gain = (
                0.1
                * self.talent_timing[agent]
                * (1 - self.state[timing_idx])
            )

            self.state[timing_idx] += gain

        elif action == 1:  # übe pitch

            gain = (
                0.1
                * self.talent_pitch[agent]
                * (1 - self.state[pitch_idx])
            )

            self.state[pitch_idx] += gain

        elif action == 2:  # gebe feedback_zum_pitch

            feedback_gain = (
                0.15
                * self.state[pitch_idx]
                * (1 - self.state[other_pitch])
            )

            self.state[other_pitch] += feedback_gain

        elif action == 3:  # gebe feedback_zum_timing

            feedback_gain = (
                0.15
                * self.state[timing_idx]
                * (1 - self.state[other_timing])
            )

            self.state[other_timing] += feedback_gain

        elif action == 4:  # wechsele rollen

            self.roles[agent], self.roles[other] = (
                self.roles[other],
                self.roles[agent]
            )

        elif action == 5:  # frage experten

            expert_pitch = 0.9
            expert_timing = 0.9

            self.state[pitch_idx] += (
                0.2 * (expert_pitch - self.state[pitch_idx])
            )

            self.state[timing_idx] += (
                0.2 * (expert_timing - self.state[timing_idx])
            )