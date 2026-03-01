from pathlib import Path

import numpy as np

from smart_vacuum.agent import QAgent, QAgentConfig
from smart_vacuum.env import SmartVacuumEnv
from smart_vacuum.training import evaluate_policy, train_agent
from smart_vacuum.visualize import visualize_policy


def main():
    """
    Orchestrates the Smart Vacuum Reinforcement Learning pipeline.

    The execution flow follows these steps:
    1.  Initialization: Instantiates the SmartVacuumEnv and QAgent
    with default configurations.
    2.  Model Persistence: Checks for a local 'q_table.npy' file.
        - If found: Loads the pre-trained weights into the agent.
        - If not: Triggers a full training session (30,000 episodes)
        and saves the resulting Q-table to disk.
    3.  Evaluation: Runs a series of greedy episodes
        to print performance metrics
        (Average Reward, Steps, and Remaining Dirt) to the console.
    4.  Visualization: Launches the interactive Pygame dashboard to watch the
        agent navigate the environment in real-time using its learned policy.
    """
    env = SmartVacuumEnv()
    agent = QAgent(QAgentConfig())

    q_path = Path(__file__).with_name("q_table.npy")

    if q_path.exists():
        agent.Q = np.load(q_path)
        print(f"Loaded existing Q-table from {q_path.name}.")
    else:
        print("Training started...")
        train_agent(env, agent, num_episodes=30000,
                    max_steps_per_episode=1000, log_interval=1000)
        np.save(q_path, agent.Q)
        print(f"Training finished and Q-table saved to {q_path.name}.")

    evaluate_policy(env, agent)

    print("Launching visualizer (greedy policy)...")
    visualize_policy(env, agent, fps=5)


if __name__ == "__main__":
    main()

