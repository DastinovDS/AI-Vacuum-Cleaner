import numpy as np


def train_agent(
    env,
    agent,
    num_episodes: int = 30000,
    max_steps_per_episode: int = 1000,
    log_interval: int = 1000,
):
    """
    Executes the Q-Learning training process over a specified number of episodes.

    This function facilitates the interaction between the agent and the environment,
    allowing the agent to learn a policy by updating its Q-table based on rewards
    received. It handles epsilon-greedy exploration decay and provides periodic
    logging of performance metrics.

    :param env: The reinforcement learning environment.
    :param agent: The agent to be trained.
    :param num_episodes: Total number of training episodes to run.
    :param max_steps_per_episode: Hard limit on steps per episode to prevent infinite loops.
    :param log_interval: How often (in episodes) to print the average reward to the console.

    :return: list[float] - A history of total rewards earned in each episode, useful for
                    plotting the learning curve.
    """
    rewards_history = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        epsilon = agent.get_epsilon(ep)

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

        if (ep + 1) % log_interval == 0:
            avg_reward = float(np.mean(rewards_history[-log_interval:]))
            print(
                f"Episode {ep + 1}/{num_episodes}, "
                f"epsilon={epsilon:.3f}, "
                f"avg_reward = {avg_reward:.2f}"
            )

    return rewards_history


def evaluate_policy(env, agent, episodes: int = 20):
    """
    Evaluates the performance of the trained agent using a purely greedy policy.

    By setting epsilon to 0.0, this function tests the 'final' learned knowledge
    of the agent without any random exploration. It collects statistics over multiple
    episodes to determine the reliability and efficiency of the vacuum's strategy.

    Metrics printed:
    - Average Reward: Overall success and penalty avoidance.
    - Average Steps: Efficiency of movement.
    - Average Remaining Dirt: Cleaning effectiveness before battery depletion.

    :param env: The reinforcement learning environment.
    :param agent: The agent whose policy is being assessed.
    :param episodes: The number of test episodes to run for averaging results.
    """
    stats = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                stats.append((total_reward, steps, info["remaining_dirt"]))
                break

    avg_reward = sum(s[0] for s in stats) / episodes
    avg_steps = sum(s[1] for s in stats) / episodes
    avg_remaining = sum(s[2] for s in stats) / episodes
    print(
        f"Eval (greedy): avg_reward={avg_reward:.1f}, "
        f"avg_steps={avg_steps:.1f}, "
        f"avg_remaining_dirt={avg_remaining:.2f}"
    )

