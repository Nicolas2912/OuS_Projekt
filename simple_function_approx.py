import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.algorithms.ppo import PPO
from ray.tune import CLIReporter
from gymnasium.wrappers import TimeLimit

from gymnasium.envs.registration import register

# Register the SinEnv environment
register(
    id='SinEnv-v0',
    entry_point='simple_function_approx:SinEnv',
    max_episode_steps=629,
    additional_wrappers=(lambda env: TimeLimit(env, max_episode_steps=629),)
)




class SinEnv(gym.Env):
    def __init__(self, env_config=None):
        super(SinEnv, self).__init__()
        self.state = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=2 * np.pi, shape=(1,))
        self.prev_target = None
        self.prev_action = None

        # self.spec = gym.spec("SinEnv-v0")
        # self.spec.max_episode_steps = 100

    def step(self, action):
        self.state += 0.01
        target = np.sin(self.state)

        # Calculate the rate of change
        target_rate_of_change = target - self.prev_target if self.prev_target is not None else 1
        action_rate_of_change = action[0] - self.prev_action if self.prev_action is not None else 1

        target = np.sin(self.state)
        prediction = np.sin(action[0])

        tolerance = 0.1
        if np.abs(target - prediction) <= tolerance:
            reward = 1
        else:
            reward = 0

        reward = -np.abs(target_rate_of_change - action_rate_of_change)

        # Update the previous target and action
        self.prev_target = target
        self.prev_action = action[0]
        done = False
        if self.state >= 2 * np.pi:
            done = True
        truncated = False

        # Print the action that the agent takes
        # print(f"Action taken by the agent: {action}")

        return np.array([self.state]), reward, done, truncated, {}

    def reset(self, **kwargs):
        self.state = 0
        # print("Resetting the environment")
        return np.array([self.state]), {}

    def __str__(self):
        return "SinEnv"

    def __repr__(self):
        return "SinEnv"


def train_agent_ppo():
    config = {
        "env": SinEnv,
        "framework": "torch",
        "num_gpus": 1,
        "num_cpus:": 20,
        "num_workers": 16,
        "lr": 0.0005,
        "horizon": 629,
        "grad_clip": 0.5,
        "soft_horizon": True,
        "batch_mode": "complete_episodes",
        "no_done_at_end": True
    }

    stop = {
        "training_iteration": 100,
    }

    # Define the reporter to print desired metrics
    reporter = CLIReporter(metric_columns=["episode_reward_mean", "policy_loss"])

    results = tune.run(PPO, config=config, stop=stop, progress_reporter=reporter, metric="episode_reward_mean",
                       mode="max",
                       checkpoint_at_end=True)

    # save agent

    return results


def train_agent_sac():
    stop = {"training_iteration": 40}
    config = {"env": SinEnv,
              "num_workers": 4,
              "gamma": 0.95,
              "lr": 0.1,
              "num_rollout_workers": 20}

    results = tune.run(SAC, config=config, stop=stop,
                       progress_reporter=CLIReporter(metric_columns=["episode_reward_mean", "policy_loss"]),
                       metric="episode_reward_mean", mode="max", checkpoint_at_end=True)

    return results


def test_agent(results):
    print(f"Results: {results}")

    # Get the trained policy
    best_trial = results.get_best_trial("episode_reward_mean", mode="max", scope="all")

    agent = PPO(config=best_trial.config, env=SinEnv)
    # genereate states to test from pi to 3*pi
    states = np.arange(0, np.pi * 2, 0.1)

    predicted_numbers = [agent.compute_single_action(np.array([state]))[0] for state in states]

    # plot real sin from pi to 3*pi (continous)
    x = np.arange(0, np.pi * 2, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

    # Predict a number for each state
    plt.scatter(states, predicted_numbers)
    plt.xlabel('State')
    plt.ylabel('Predicted Number')
    plt.title('Predictions made by the agent')
    plt.show()


results = train_agent_ppo()
test_agent(results)
# ray.rllib.utils.check_env(SinEnv())
