from dataclasses import dataclass

import numpy as np


@dataclass
class QAgentConfig:
    """
    Configuration parameters for the Q-Learning Agent.

    :ivar num_states: Dimension of the observation space (size of Q-table rows).
    :ivar num_actions: Number of possible actions (size of Q-table columns).
    :ivar alpha: Learning rate, determining how much new information overrides old.
    :ivar gamma: Discount factor, determining the importance of future rewards.
    :ivar epsilon_start: Initial exploration probability.
    :ivar epsilon_end: Minimum exploration probability after decay.
    :ivar epsilon_decay_episodes: Number of episodes over which epsilon decreases.
    """
    num_states: int = 256
    num_actions: int = 5  # N, S, W, E, Clean
    alpha: float = 0.05
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 25000


class QAgent:
    """
    A reinforcement learning agent using the Q-Learning (Temporal Difference) algorithm.

    This agent maintains a Q-table to estimate the expected future rewards for every
    state-action pair. It uses an epsilon-greedy strategy to balance exploration
    of the environment with exploitation of its learned knowledge.
    """
    def __init__(self, config: QAgentConfig):
        self.config = config
        self.Q = np.zeros((config.num_states, config.num_actions), dtype=np.float32)

    def get_epsilon(self, episode_index: int) -> float:
        """
        Calculates the current exploration rate (epsilon) using linear decay.

        As training progresses, the likelihood of taking random actions decreases,
        allowing the agent to rely more on its learned Q-values.

        :param episode_index: The index of the current training episode.

        :return: float - The probability of taking a random action.
        """
        c = self.config
        if episode_index >= c.epsilon_decay_episodes:
            return c.epsilon_end
        fraction = episode_index / c.epsilon_decay_episodes
        return c.epsilon_start + fraction * (c.epsilon_end - c.epsilon_start)

    def select_action(self, state: int, epsilon: float) -> int:
        """
        Chooses an action based on the epsilon-greedy policy for a given state.

        With probability epsilon, the agent explores (chooses a random action).
        Otherwise, it exploits (chooses the action with the highest Q-value).
        If multiple actions share the same maximum Q-value, one is chosen at random
        to prevent deterministic loops.

        :param state: The current encoded state index.
        :param epsilon: The current exploration probability.

        :return: int - The index of the chosen action.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.config.num_actions))

        # Greedy action with random tie-breaking (prevents deterministic loops)
        row = self.Q[state]
        max_q = float(np.max(row))
        best_actions = np.flatnonzero(row == max_q)
        return int(np.random.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Updates the Q-table entry using the Bellman Equation (TD Learning).

        This function calculates the Temporal Difference (TD) target and error
        to adjust the Q-value for the state-action pair just executed. It incorporates
        the immediate reward and the discounted maximum future reward of the next state.

        :param state: The current state index.
        :param action: The action performed in state s.
        :param reward: The reward received after performing action a.
        :param next_state: The resulting next state index.
        :param done: Whether the episode terminated after this step.
        """
        alpha = self.config.alpha
        gamma = self.config.gamma

        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = reward + gamma * best_next
        td_error = td_target - float(self.Q[state, action])
        self.Q[state, action] += alpha * td_error

