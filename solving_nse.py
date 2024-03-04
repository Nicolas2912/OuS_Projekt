import gymnasium as gym
import random
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, TD3, SAC
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
import structlog

import torch

print(f"CUDA: ", torch.cuda.is_available())

log_dir = "./tmp/"
os.makedirs(log_dir, exist_ok=True)

logger = structlog.get_logger()

def distance_func_continuous(point, func):
    point = np.array(point)

    # Define a function that calculates the distance between a point and the function
    def distance_point_to_function_helper(x):
        # Calculate the distance between the point and the function at point x
        return np.linalg.norm(point - np.array([x, func(x)]))

    # Use scipy.optimize to find the minimum distance
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(distance_point_to_function_helper)

    # Return the minimum distance found
    return result.fun


def distance_func_discrete(point, x_array, y_array):
    all_points = np.vstack((x_array, y_array)).T
    distances = np.sqrt(np.sum((all_points - np.array([point[0], point[1]])) ** 2, axis=1))

    return np.min(distances)


def discrete_nse(x_min, x_max, num_points=100000):
    eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
    eq2 = lambda x: np.sin(2*x)

    x = np.linspace(x_min, x_max, num_points)
    # x = x[x != 0]

    yeq1 = np.vectorize(eq1)
    yeq2 = np.vectorize(eq2)

    y_arrayeq1 = yeq1(x)
    y_arrayeq2 = yeq2(x)

    return np.array([y_arrayeq1, y_arrayeq2]), x


def nse():
    """
    System of nonlinear equations.
    :return: Numpy array of equations.
    """
    eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
    eq2 = lambda x: np.sin(2*x)
    return np.array([eq1, eq2])


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


def plot_nse(equations, x_array, points=None):
    for eq in equations:
        y = eq(x_array)
        plt.plot(x_array, y)
    if points is not None:
        for point in points:
            plt.scatter(point[0], point[1], color='red')

    plt.xlim(-1.5, 3.0)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.show()


def plot_discrete_nse(y_array, x_min, x_max, y_min, y_max, num_points=1000, points=[]):
    x = np.linspace(x_min, x_max, num_points)
    # x = x[x != 0]

    plt.plot(x, y_array[0])
    plt.plot(x, y_array[1])

    plt.ylim(y_min, y_max)

    if len(points) > 0:
        for point in points:
            plt.scatter(point[0], point[1], color='red')

    plt.grid()
    plt.show()


class SaveActionsCallback(BaseCallback):
    def __init__(self, check_freq: int, actions_list: list):
        super(SaveActionsCallback, self).__init__()
        self.check_freq = check_freq
        self.actions_list = actions_list

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            action, _ = self.model.predict(self.training_env.envs[0].last_obs, deterministic=True)
            self.actions_list.append(action)
        return True


class CustomEnv(gym.Env):
    def __init__(self, nse, plot=False, discrete=False):
        super(CustomEnv, self).__init__()
        self.x_min = -10.0
        self.x_max = 10.0
        self.y_min = -10.0
        self.y_max = 10.0
        self.action_space = gym.spaces.Box(low=np.array([self.x_min, self.y_min]),
                                           high=np.array([self.x_max, self.y_max]),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1000.0, high=1000.0, shape=(2,))
        self.state = np.array(
            [random.uniform(self.x_min, self.x_max),
             random.uniform(self.y_min, self.y_max)])  # self.state = point in space
        self.nse = nse
        self.nse_discrete, self.x_array = discrete_nse(self.x_min, self.x_max)
        self.best_action = []
        self.last_obs = None
        self.actions = []
        self.distances = []
        self.best_distances = []
        self.good_points = []

        if plot:
            plot_nse(self.nse, self.x_array)

    def get_distance(self, point):
        """
        Calculate the distance between a point and each equation in the system of nonlinear equations.
        :param point: Tuple containing the coordinates of the point.
        :return: Numpy array of distances between the point and each equation.
        """
        return np.array([distance_func_continuous(point, eq) for eq in self.nse])

    def get_distance_discrete(self, point):
        return np.array([distance_func_discrete(point, self.x_array, eq) for eq in self.nse_discrete])

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action to take. (Tuple containing the coordinates of the point)
        :return: state, reward, done, info
        """

        discrete = True
        self.actions.append(action)
        if discrete:
            self.distances.append(sum(self.get_distance_discrete(action)))
        else:
            self.distances.append(sum(self.get_distance(action)))

        good_points_threshold = 0.1
        # just print better action
        if self.distances[-1] <= min(self.distances):
            print("Best action:", self.actions[-1])
            print("Distance:", self.distances[-1])
            print()
            self.best_action.append(action)
            self.best_distances.append(self.distances[-1])

        if self.distances[-1] <= good_points_threshold and len(self.good_points) > 0:
            print(f"Good point: {self.good_points[-1]}\tDistance: {self.distances[-1]}")

        if discrete:
            self.state = action
            distances = self.get_distance_discrete(action)

            if np.sum(distances) <= good_points_threshold:
                self.good_points.append(action)

            reward = np.sum(-distances)
            done = False
            truncated = False
            if np.sum(distances) <= 1.0:  # if distance is less than reset
                done = True
            # self.best_action.append(action)
            self.last_obs = self.state

            return self.state, reward, done, truncated, {}
        else:
            # reward negative distance
            self.state = action
            distances = self.get_distance(action)

            if np.sum(distances) <= good_points_threshold:
                self.good_points.append(action)

            reward = np.sum(-distances)
            done = False
            truncated = False
            if np.sum(distances) <= 1.0:  # if distance is less than reset
                done = True

            # self.best_action.append(action)
            self.last_obs = self.state

            # print(self.best_action[-1])

            return self.state, reward, done, truncated, {}

    def reset(self, seed=None):
        """
        Reset the state of the environment and return an initial observation.
        :param seed: Seed must be set
        :return: state, {}
        """
        discrete = True

        self.state = np.array([random.uniform(self.x_min, self.y_min), random.uniform(self.x_max, self.y_max)],
                              dtype=np.float32)  # Update to have two points
        # sort self.best_action
        if len(self.best_action) > 0 and not discrete:
            distance_action_mapping = [(action, np.sum(self.get_distance(action))) for action in self.best_action]
            # sort by distance ascending
            best = min(distance_action_mapping, key=lambda x: x[1])
            # print("Best action:", best[0], "Distance:", best[1])
            # print first action with lowest distance

        if len(self.best_action) > 0 and discrete:
            distance_action_mapping = [(action, np.sum(self.get_distance_discrete(action))) for action in
                                       self.best_action]
            # sort by distance ascending
            best = min(distance_action_mapping, key=lambda x: x[1])
            # print("Best action:", best[0], "Distance:", best[1])
            # print first action with lowest distance

        # self.best_action = []
        self.last_obs = self.state

        return self.state, {}


if __name__ == '__main__':
    # --- plot discrete nse ---
    # equations = discrete_nse(-10.0, 8.0, -10.0, 10.0)
    # plot_discrete_nse(equations, -10.0, 8.0, -10.0, 10.0)

    # init environment
    env = CustomEnv(nse(), plot=False, discrete=True)
    logger.info("Environment created")

    # check environment
    # check_env(env)

    # create model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, ent_coef=0.15)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
                  buffer_size=1000)
    logger.info("Model created")
    # model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,)
    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.15)

    # train model
    log = False
    if log:
        actions = []
        callback = SaveActionsCallback(1, actions)
        logger.info("Start training")
        model.learn(total_timesteps=int(5e3), progress_bar=False, tb_log_name="DDPG_NSE")
    else:
        actions = []
        callback = SaveActionsCallback(1, actions)
        logger.info("Start training")
        model.learn(total_timesteps=int(3e3), progress_bar=False)

    actions, distances = env.best_action, env.distances
    good_points = env.good_points

    print(f"Number of good points: {len(good_points)}")
    # print(f"Good points: {good_points}")

    x_min = -3.0
    x_max = 3.0
    y_min = -3.0
    y_max = 3.0
    plot_nse(nse(), np.linspace(x_min, x_max, 100000), points=good_points)

    # save model
    model.save("ddpg_nse")

    # tensorboard befehl: tensorboard --logdir ./tmp/
