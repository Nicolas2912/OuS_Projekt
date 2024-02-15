import numpy as np
import gymnasium as gym
import random
import time
import matplotlib.pyplot as plt
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env


def nse():
    eq1 = lambda x, y: x**2 + y**2 - 1
    eq2 = lambda x, y: x - y + 1
    return np.array([eq1, eq2])

def nse_eval(equations, x, y):
    return np.array([eq(x, y) for eq in equations])

def plot_nse(equations):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = equations[0](X, Y)
    Z2 = equations[1](X, Y)
    plt.contour(X, Y, Z1, [0], colors='r')
    plt.contour(X, Y, Z2, [0], colors='b')
    plt.show()

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,))
        self.state = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])


    def step(self, action):
        self.state += action
        reward = -np.linalg.norm(nse_eval(equations, *self.state))
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        return self.state


if __name__ == '__main__':
    equations = nse()
    plot_nse(equations)