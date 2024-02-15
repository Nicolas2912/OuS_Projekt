import numpy as np
import gymnasium as gym
import random
import numpy as np
import numdifftools as nd
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

log_dir = "./tmp/"
os.makedirs(log_dir, exist_ok=True)
from stable_baselines3.common.env_checker import check_env


def distance_func(point, func, *args):
    """
    Calculate the distance between a point and a function.

    Parameters:
        point (tuple): A tuple containing the coordinates of the point.
        func (callable): The function to calculate the distance to.
        *args: Additional arguments to pass to the function.

    Returns:
        float: The distance between the point and the function.
    """
    return abs(func(*point, *args))


def nse():
    """
    System of nonlinear equations.
    :return: Numpy array of equations.
    """
    eq1 = lambda x, y: x ** 2 + y ** 2 - 1
    eq2 = lambda x, y: x - y + 1
    return np.array([eq1, eq2])


def get_overall_distance(point, nse):
    return np.sum([distance_func(point, eq) for eq in nse])


def plot_function_and_point(func, point, closet_point, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5)):
    """
    Plot the given function along with a point on it.

    Parameters:
        func (callable): The function to plot.
        point (tuple): A tuple containing the coordinates of the point.
        xlim (tuple): A tuple containing the limits for the x-axis.
        ylim (tuple): A tuple containing the limits for the y-axis.
    """
    # Create a meshgrid for plotting the function
    x_values = np.linspace(xlim[0], xlim[1], 1000)
    y_values = np.linspace(ylim[0], ylim[1], 1000)
    X, Y = np.meshgrid(x_values, y_values)
    Z = func(X, Y)

    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=[0], colors='blue')  # Plot the contour for the circle equation

    # Plot the point
    plt.scatter(point[0], point[1], color='red', label='Point')

    # Plot the closest point
    if closet_point is not None:
        plt.scatter(closet_point[0], closet_point[1], color='green', label='Closest Point')

    # Set labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function and Point Plot')

    # Set limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Show legend and grid
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal for a proper circle
    # plt.show()


def plot_nse(equations):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = equations[0](X, Y)
    Z2 = equations[1](X, Y)
    plt.contour(X, Y, Z1, [0], colors='r')
    plt.contour(X, Y, Z2, [0], colors='b')
    plt.grid()
    plt.show()


class CustomEnv(gym.Env):
    def __init__(self, nse):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(2,))
        # self.state = self.get_distance(self.action_space.sample())
        self.observation_space = gym.spaces.Box(low=-1000.0, high=1000.0, shape=(2,))
        self.state = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])  # self.state = point in space
        self.nse = nse
        self.best_action = []

    def get_distance(self, point):
        """
        Calculate the distance between a point and each equation in the system of nonlinear equations.
        :param point: Tuple containing the coordinates of the point.
        :return: Numpy array of distances between the point and each equation.
        """
        return np.array([distance_func(point, eq) for eq in self.nse])

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action to take. (Tuple containing the coordinates of the point)
        :return: state, reward, done, info
        """
        # reward negative distance
        self.state = action
        distances = self.get_distance(action)

        reward = np.sum(-distances)
        done = False
        truncated = False
        if np.sum(distances) <= 0.005:  # if distance is less than 0.005. Can be adjustet
            done = True

        self.best_action.append(action)

        return self.state, reward, done, truncated, {}

    def reset(self, seed=None):
        """
        Reset the state of the environment and return an initial observation.
        :param seed: Seed must be set
        :return: state, {}
        """
        self.state = np.array([random.uniform(-10, 10), random.uniform(-10, 10)],
                              dtype=np.float32)  # Update to have two points
        # sort self.best_action
        if len(self.best_action) > 0:
            distance_action_mapping = [(action, np.sum(self.get_distance(action))) for action in self.best_action]
            # sort by distance ascending
            best = min(distance_action_mapping, key=lambda x: x[1])
            print("Best action:", best[0], "Distance:", best[1])
            # print first action with lowest distance

        self.best_action = []

        return self.state, {}


if __name__ == '__main__':

    # init environment
    env = CustomEnv(nse())

    # check environment
    # check_env(env)

    # create model
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1, tensorboard_log=log_dir)

    # train model
    model.learn(total_timesteps=int(2e5), progress_bar=False, tb_log_name="PPO_NSE")

    # save model
    model.save("ppo_nse")

    # plot results
    results_plotter.plot_results([log_dir], int(2e5), results_plotter.X_TIMESTEPS, "PPO NSE")

    # Display the plot
    plt.show()
    # --- testing ---

    # load model
    model = PPO.load("ppo_nse", env=env)

    observation, _ = env.reset()  # Only take the observation part of the tuple
    print(f"Observation: {observation}")

    for i in range(25):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        print("Action:", action)
        print(f"Distance: {np.sum(env.get_distance(action))}")

        if done:
            print("Episode finished after {} timesteps".format(i + 1))
            break

