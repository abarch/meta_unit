import numpy as np
import matplotlib.pyplot as plt
import random


class DynaQAgent:
    """
    A Dyna-Q agent for reinforcement learning that combines Q-learning
    with planning.

    Attributes:
    ----------

    n_states (int): Number of states in the environment.
    n_actions (int): Number of possible actions in the environment.
    epsilon (float): Probability of choosing a random action (exploration).
    alpha (float): Learning rate for updating Q-values.
    gamma (float): Discount factor for future rewards.
    planning_steps (int): Number of planning steps to perform.
    q_table (ndarray): Q-values table for state-action pairs.
    model (dict): Model to store state transitions and rewards.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.95,
        planning_steps: int = 5,
    ) -> None:
        """
        Initialize the DynaQAgent with the given parameters.

        Args:
        n_states (int): Number of states in the environment.
        n_actions (int): Number of possible actions in the environment.
        epsilon (float): Probability of choosing a random action (exploration).
        alpha (float): Learning rate for updating Q-values.
        gamma (float): Discount factor for future rewards.
        planning_steps (int): Number of planning steps to perform.

        Returns:
        None
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.planning_steps = planning_steps

        # Initialize Q-table and model
        self.q_table = np.zeros((n_states, n_actions))
        self.model = {}

    def choose_action(self, state: int) -> int:
        """Choose an action.

        Choose an action based on the current state using an epsilon-greedy
        policy.

        Args:
        state (int): The current state of the environment.

        Returns:
        int: The action chosen, either randomly (exploration) or based on the
        highest Q-value (exploitation).
        """
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the Q-table and model with the given transition and perform
        planning steps.

        Args:
        state (int): The current state of the environment.
        action (int): The action taken from the current state.
        reward (float): The reward received after taking the action.
        next_state (int): The state transitioned to after taking the action.

        Returns:
        None
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

        # Store the transition in the model
        self.model[(state, action)] = (reward, next_state)

        # Planning phase
        for _ in range(self.planning_steps):
            s, a = random.choice(list(self.model.keys()))
            r, s_next = self.model[(s, a)]
            best_next_a = np.argmax(self.q_table[s_next])
            td_target = r + self.gamma * self.q_table[s_next][best_next_a]
            td_error = td_target - self.q_table[s][a]
            self.q_table[s][a] += self.alpha * td_error


# Model the learner that peforms an action and does not get a reward any time that the action is performed (while assumsing that the policy of the q-learner is not optimal.
class SimulatedHumanLearner:
    """Simulated Human Learner."""

    def __init__(
        self,
        n_states: int,
        error_probability: float,
        smart_action_sequence_learner: list[int],
    ) -> None:
        """Initialize the agent with the given parameters.

        Args:
        n_states (int): Number of states in the environment.
        error_probability (float): Probability of making an error.
        smart_action_sequence_learner (list[int]): Learner for smart action
        sequences.

        Returns:
        None
        """
        self.state = 0
        self.n_states = n_states
        self.error_probability = error_probability
        self.smart_action_sequence_learner = smart_action_sequence_learner

    def perform_action(self, action):
        # The learner may progress or regress depending on action and error probability
        if random.uniform(0, 1) < self.error_probability:
            self.state = max(0, self.state)
            return 0  # Error made
        elif action == self.smart_action_sequence_learner[self.state]:
            self.state = min(self.n_states - 1, self.state + 1)
            return 1  # No error
        return 0

    def get_state(self):
        return self.state


class SimulatedTeacher:
    def __init__(self, smart_action_sequence):
        self.smart_action_sequence = smart_action_sequence

    def should_intervene(self, avg_reward):
        # Inverse the average reward to determine the probability of intervention
        teacher_peak = 1 - avg_reward
        return np.random.binomial(1, teacher_peak) == 1

    def guide(self, human_learner):
        current_state = human_learner.get_state()
        if current_state < len(self.smart_action_sequence):
            action = self.smart_action_sequence[current_state]
            # Teacher ensures the learner progresses according to the smart action sequence
            human_learner.state = min(
                human_learner.n_states - 1, human_learner.state + 1
            )
            reward = 1
            return (
                current_state,
                action,
                reward,
                human_learner.get_state(),
            )  # No error, as the teacher is guiding correctly
        else:
            print("weird case that should not take place")
            # return 0  # No guidance needed if beyond the smart action sequence


def simulate_learning(agent, human_learner, teacher, teacherF, episodes=100):
    rewards = []

    steps = 0
    while (human_learner.get_state() < human_learner.n_states - 1) or (
        steps < episodes
    ):
        # for episode in range(episodes):

        # Calculate the average reward (error rate) over the last 10 episodes
        avg_reward = (
            np.mean(rewards[-10:]) if len(rewards) >= 10 else 1
        )  # Use 1 if insufficient history

        # Decide whether the teacher should intervene based on a Bernoulli process
        if teacher.should_intervene(avg_reward) and teacherF == 1:
            (state, action, reward, next_state) = teacher.guide(
                human_learner
            )  # Teacher intervenes
        elif teacherF == 0.5:
            threshold = random.uniform(0, 1)

            if threshold < 0.2:
                (state, action, reward, next_state) = teacher.guide(human_learner)
            else:
                state = human_learner.get_state()
                action = agent.choose_action(state)
                reward = human_learner.perform_action(
                    action
                )  # Learner performs action based on Q-learner's decision
                next_state = human_learner.get_state()

        else:
            state = human_learner.get_state()
            action = agent.choose_action(state)
            reward = human_learner.perform_action(
                action
            )  # Learner performs action based on Q-learner's decision
            next_state = human_learner.get_state()
        agent.update(state, action, reward, next_state)
        rewards.append(reward)
        # print (teacher_flag, learner_flag, " :: ", state, action, reward, next_state)
        steps += 1

    avg_reward = np.mean(rewards)
    # print (human_learner.get_state())
    return avg_reward


#############
# episodes == number of evaluations
def evaluate_policy(agent, human_learner, episodes=10):
    evaluation_rewards = []

    # Turn off exploration by setting epsilon to 0
    agent.epsilon = 0.2

    for episode in range(episodes):
        episode_rewards = 0
        human_learner.state = 0  # Reset the learner to the initial state
        steps = 0
        while human_learner.get_state() < human_learner.n_states - 1:
            state = human_learner.get_state()
            action = agent.choose_action(
                state
            )  # Choose the best action based on the trained Q-table

            reward = human_learner.perform_action(action)
            next_state = human_learner.get_state()
            episode_rewards += reward
            # print (state, action, reward, next_state)
            steps += 1
            # Break the loop if the learner reaches the final state
            if next_state == human_learner.n_states - 1:
                break

        evaluation_rewards.append(episode_rewards / steps)
        # print(f"Episode {episode+1}: Total Reward = {episode_rewards/steps}")

    avg_reward = np.mean(evaluation_rewards)
    # print(f"Average Reward over {episodes} evaluation episodes: {avg_reward}")

    return avg_reward


# simulation and eval
################

# Simulation parameters
n_states = 24  # Number of states (learner's capabilities)
n_actions = 4  # Number of possible exercise units
epsilon = 0.1  # Exploration factor
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
planning_steps = (
    5  # Number of planning steps for Dyna-Q (10 steps, 0.3 exploration during test)
)
error_probability = 0.5  # Probability that the learner makes an error

# Define the smart action sequence known to the teacher
smart_action_sequence = [0, 1, 2, 3] * (n_states // n_actions)
smart_action_sequence_learner = [1, 1, 2, 3] * (n_states // n_actions)
# print (len(smart_action_sequence))
# Instantiate the learner, agent, and teacher

training_iterations = 100


def eval(training_iterations):
    # Simulate the learning process
    x = []
    y = []
    agent_1 = DynaQAgent(n_states, n_actions, epsilon, alpha, gamma, planning_steps)
    agent_2 = DynaQAgent(n_states, n_actions, epsilon, alpha, gamma, planning_steps)
    teacher = SimulatedTeacher(smart_action_sequence)

    for _ in range(training_iterations):
        human_learner_1 = SimulatedHumanLearner(
            n_states, error_probability, smart_action_sequence_learner
        )
        # print ("initial state", human_learner.get_state())
        av_reward_meta = simulate_learning(
            agent_1, human_learner_1, teacher, 1, episodes=30
        )
        x.append(av_reward_meta)
        human_learner_2 = SimulatedHumanLearner(
            n_states, error_probability, smart_action_sequence_learner
        )
        av_reward_q_only = simulate_learning(
            agent_2, human_learner_2, teacher, 0, episodes=30
        )
        # av_reward_q_after_training = evaluate_policy(agent, human_learner, teacher, episodes=1)
        y.append(av_reward_q_only)
    return x, y


def run(average_iterations):
    x = []
    y = []

    for _ in range(average_iterations):
        meta, q = eval(training_iterations)
        # print (x,y)
        x.append(meta)
        y.append(q)

    stacked_x = np.stack(x)
    stacked_y = np.stack(y)

    meta_m, meta_v = mean_var(stacked_x)
    q_m, q_v = mean_var(stacked_y)
    return meta_m, meta_v, q_m, q_v


def mean_var(stacked):
    mean = np.mean(stacked, axis=0)
    variance = np.var(stacked, axis=0)
    return mean, variance


average_iterations = 40
mean_x, variance_x, mean_y, variance_y = run(average_iterations)


slice = 5
# plt.scatter(x, y)
# plt.show()
# learner_error = 0.4
plt.errorbar(
    np.arange(training_iterations)[::slice],
    mean_x[::slice],
    yerr=variance_x[::slice],
    fmt="o-",
    capsize=5,
    label="Mu/Sigma (Meta Unit Training:Teacher(Bernoulli) + Q-learner)",
    color="grey",
)
# plt.errorbar(np.arange(training_iterations)[::slice], mean_y[::slice], yerr=variance_y[::slice], fmt='o-', capsize=5, label='Mu/Sigma (Meta Unit Training: Teacher 20% + Q-learner)', color="blue")
plt.errorbar(
    np.arange(training_iterations)[::slice],
    mean_y[::slice],
    yerr=variance_y[::slice],
    fmt="o-",
    capsize=5,
    label="Mu/Sigma (Q-learner Training)",
    color="blue",
)
# plt.plot(x, label="Training (with teacher)", color="grey")
# plt.plot(y, label="Evaluation (without teacher)", color="blue")
# plt.xlabel("Number of training episodes (agent)")
# plt.ylabel("Improvement rate")
plt.title("Meta unit training (grey) and Q-learner evaluation (blue)")
# plt.title("Meta Unit training (Bernoulli) (grey), Meta Unit training with 20% teacher interventions(blue)")
plt.legend()
plt.show()
