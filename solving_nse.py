import gymnasium as gym
import random
import math
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


def plot_nse(equations, x_array, dimension=1, contour_plot=False, points=None, points_x=None, distances=None):
    if dimension == 1:
        fig, axs = plt.subplots(1, 2, figsize=(15, 9))
        for eq in equations:
            y = eq(x_array)
            axs[0].plot(x_array, y)
        if points is not None:
            for point in points:
                axs[0].scatter(point[0], point[1], color='red')
        if points_x is not None:
            for x_value in points_x:
                axs[0].axvline(x=x_value, color='red')

        if distances is not None:
            axs[1].plot(distances, marker='o')

        axs[0].grid()
        axs[1].set_title("Best Residuen")
        axs[1].set_yscale('log')
        axs[1].grid()

        axs[0].set_ylim(-3.0, 3.0)
        axs[0].set_xlim(-3.0, 3.0)
        plt.show()


    elif dimension == 2:
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        points_array = np.stack((X, Y), axis=-1)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sum(eq(points_array[i, j]) for eq in equations)

        if contour_plot:
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, Z, cmap='viridis')
            plt.colorbar(label='NSE')
            if points is not None:

                for i, point in enumerate(points):
                    # plot points but just one label
                    if i == 0:
                        plt.scatter(point[0], point[1], color='red', label='Points')
                    else:
                        plt.scatter(point[0], point[1], color='red')

            # plot real minimum at (1,1) in blue
            plt.scatter(1, 1, color='blue', label='Minimum')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Contour Plot of the NSE')
            plt.grid()
            plt.legend()
            plt.show()

        else:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)

            if points is not None:
                for i, point in enumerate(points):
                    # Provide the z-coordinate as 0 or any other constant value
                    z_coord = 0
                    if i == 0:
                        ax.scatter(point[0], point[1], z_coord, color='red', label='Points')
                    else:
                        ax.scatter(point[0], point[1], z_coord, color='red')

            # plot minimum at (1,1) in blue
            ax.scatter(1, 1, 0, color='blue', label='Minimum')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('NSE')
            ax.set_title('NSE of the 2D Function')
            ax.legend()
            plt.tight_layout()
            plt.show()

    else:
        raise ValueError(f"Plotting for dimension {dimension} is not supported.")


def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


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
    def __init__(self, dimension):
        super(CustomEnv, self).__init__()
        self.x_min = -10.0
        self.x_max = 10.0
        self.y_min = -5.0
        self.y_max = 5.0

        self.dimension = dimension
        self.low_bounds = -5.0 * np.ones(dimension)
        self.high_bounds = 5.0 * np.ones(dimension)
        self.action_space = gym.spaces.Box(low=self.low_bounds,
                                           high=self.high_bounds,
                                           dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(2,))
        self.state = np.array(
            [random.uniform(self.x_min, self.x_max),
             random.uniform(self.y_min, self.y_max)])  # self.state = point in space
        self.best_actions = []
        self.last_obs = None
        self.actions = []
        self.distances = []
        self.best_distances = []
        self.good_points = []
        self.good_points_plot = []
        self.good_res_plot = []
        self.best_action_so_far = None
        self.good_points_thrs = 0.1
        self.rewards = []
        self.reward_action_map = []

        self.consecutive_no_improvement = 0
        self.total_steps = 0

        self.improvement_rate_window_size = 10
        self.improvement_rate_history = []

        self.action_precision = dict()
        self.time_residuum = list()
        self.time_action = list()

        self.min_residuum = float('inf')
        self.max_residuum = float('-inf')

        self.best_residuum = float('inf')

        self.start_time = time.time()

    def calculate_derivative(self, action):
        d_eq1 = lambda x: 5 * x ** 4 - 12 * x ** 3 + 3 * x ** 2 + x
        d_eq2 = lambda x: 2 * np.cos(2 * x)

        return np.array([d_eq1(action), d_eq2(action)])

    def nse(self):
        # eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
        # eq2 = lambda x: np.sin(2 * x)

        rosenbrock_eq1 = lambda x: 10 * (x[1] - x[0] ** 2)
        rosenbrock_eq2 = lambda x: 1 - x[0]

        eq1 = lambda x: np.sin(x) ** 3 - 2 * np.exp(x) + x ** 2 - 4 * x + 2
        eq2 = lambda x: np.exp(-x) * np.sin(3 * x)
        # eq3 = lambda x: x ** 2 * np.sin(x) - np.exp(-x)
        eq4 = lambda x: np.sin(x) ** 2 - x ** 3 * np.exp(-x)
        eq5 = lambda x: np.exp(x / 2) * np.sin(2 * x) - x ** 2

        return np.array([rosenbrock_eq1, rosenbrock_eq2])

    def get_distance_discrete(self, point):
        """
        Calculate the residuum for the given point.
        :param point:
        :param nse:
        :return:
        """
        use_log_residuum = True

        equations = self.nse()

        # Initialize an empty list to store the values of each equation
        values = []

        # Iterate over each equation in the nse
        for eq in equations:
            # Calculate the function values
            value = eq(point)
            values.append(value)

        if use_log_residuum:
            # Logarithmic residuum
            log_base = 10  # Basis of the logarithm (adjustable)

            # Initialize an empty list to store the scaled values of each equation
            scaled_values = []

            for value in values:
                if isinstance(value, np.ndarray):
                    scaled_value = np.log(np.abs(value) + 1) / np.log(log_base)
                    scaled_value[value < 0] = -scaled_value[value < 0]
                else:
                    scaled_value = np.log(abs(value) + 1) / np.log(log_base)
                    if value < 0:
                        scaled_value = -scaled_value

                scaled_values.append(scaled_value)

            res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                      range(i + 1, len(scaled_values)))
        else:
            # Normal residuum
            res = sum((values[i] - values[j]) ** 2 for i in range(len(values)) for j in range(i + 1, len(values)))

        return res

    def _plot_res(self, dimension, plot_contour=False):
        num_points = 1000

        if dimension == 1:
            # 1-dimensional function
            x = np.linspace(-10, 10, num_points)
            residuals = np.array([self.get_distance_discrete(np.array([point])) for point in x])

            plt.figure(figsize=(8, 6))
            plt.plot(x, residuals)
            plt.xlabel('x')
            plt.ylabel('Residuum')
            plt.title('Residuum of the 1D Function')
            plt.grid()
            plt.show()

        elif dimension == 2:
            # 2-dimensional function (e.g., Rosenbrock function)
            x = np.linspace(-10, 10, num_points)
            y = np.linspace(-10, 10, num_points)
            X, Y = np.meshgrid(x, y)
            points = np.stack((X, Y), axis=-1)

            residuals = np.array([self.get_distance_discrete(point) for point in points.reshape(-1, 2)])
            Z = residuals.reshape(X.shape)

            if plot_contour:
                plt.figure(figsize=(8, 6))
                plt.contour(X, Y, Z, levels=50)

                # make point at (1,1)
                plt.scatter(1, 1, color='red', label='Minimum')

                plt.colorbar(label='Residuum')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Contour Plot of the Residuum')
                plt.grid()
                plt.show()
            else:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('Residuum')
                ax.set_title('Residuum of the 2D Function')
                plt.tight_layout()
                plt.show()

        else:
            raise ValueError(f"Plotting for dimension {dimension} is not supported.")

        # Logarithmische Skalierung der Funktionswerte
        # log_base = 5  # Basis des Logarithmus (anpassbar)
        # scaled_eq1 = np.log(abs(value_eq1) + 1) / np.log(log_base)
        # scaled_eq2 = np.log(abs(value_eq2) + 1) / np.log(log_base)
        #
        # # Vorzeichenkorrektur für negative Werte
        # scaled_eq1[value_eq1 < 0] = -scaled_eq1[value_eq1 < 0]
        # scaled_eq2[value_eq2 < 0] = -scaled_eq2[value_eq2 < 0]
        #
        # res = (scaled_eq1 - scaled_eq2) ** 2
        #
        # res_old = (eq1(x) - eq2(x)) ** 2

        # plot res and res old
        # plt.plot(x, res, label="res log")
        # plt.plot(x, res_old, label="res normal")
        # plt.legend()
        # plt.grid()
        # plt.show()

    def logarithmic_reward(self, residuum, min_residuum, max_residuum, alpha=1.0, beta=1.0):
        """
        Logarithmic reward function that provides more informative rewards in high residuum regions.

        Args:
            residuum (float): Current residuum value.
            min_residuum (float): Minimum residuum value encountered so far.
            max_residuum (float): Maximum residuum value encountered so far.
            alpha (float): Scaling factor for the logarithmic term.
            beta (float): Scaling factor for the linear term.

        Returns:
            float: Reward value.
        """
        if max_residuum - min_residuum != 0:
            normalized_residuum = (residuum - min_residuum) / (max_residuum - min_residuum)
        else:
            normalized_residuum = 0

        logarithmic_term = -alpha * np.log(residuum + 1e-8)  # Logarithmic term with a small constant to avoid log(0)
        linear_term = -beta * normalized_residuum  # Linear term for residuum normalization

        reward = logarithmic_term + linear_term

        return reward

    def adaptive_reward(self, residuum, min_residuum, max_residuum, reward_type='logarithmic', alpha_init=1.0,
                        beta_init=1.0,
                        gamma=0.1, state=None, decay_rate=0.99):
        """
        Adaptive reward function that provides different reward types (logarithmic or exponential) and includes gradient information.

        Args:
            residuum (float): Current residuum value.
            min_residuum (float): Minimum residuum value encountered so far.
            max_residuum (float): Maximum residuum value encountered so far.
            reward_type (str): Type of reward function to use ('logarithmic' or 'exponential').
            alpha (float): Scaling factor for the reward term.
            beta (float): Scaling factor for the normalization term.
            gamma (float): Scaling factor for the gradient term.
            state (tuple): Current state (x, y) coordinates.

        Returns:
            float: Reward value.
        """
        if max_residuum - min_residuum != 0:
            normalized_residuum = (residuum - min_residuum) / (max_residuum - min_residuum)
        else:
            normalized_residuum = 0

        # Update the scaling factors based on the current progress
        alpha = alpha_init * decay_rate ** len(self.rewards)
        beta = beta_init * decay_rate ** len(self.rewards)

        if reward_type == 'logarithmic':
            reward_term = -alpha * np.log(residuum + 1e-8)
        elif reward_type == 'exponential':
            reward_term = -alpha * np.exp(residuum)
        else:
            raise ValueError(f"Invalid reward type: {reward_type}. Choose 'logarithmic' or 'exponential'.")

        normalization_term = -beta * normalized_residuum

        if state is not None:
            x, y = state
            gradient = rosenbrock_gradient(x, y)
            gradient_term = -gamma * np.linalg.norm(gradient)
        else:
            gradient_term = 0

        reward = reward_term + normalization_term + gradient_term

        return reward

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action to take. (Tuple containing the coordinates of the point)
        :return: state, reward, done, info
        """

        discrete = True
        self.actions.append(action)
        residuum = self.get_distance_discrete(action)
        self.distances.append(residuum)

        # just print better action
        if self.distances[-1] <= min(self.distances):
            print("Best action:", self.actions[-1])
            print("Residuum:", self.distances[-1])
            # print()
            self.best_actions.append(action)
            self.best_distances.append(self.distances[-1])

        if self.distances[-1] <= self.good_points_thrs and len(self.good_points) > 0:
            # print(f"Good point: {self.good_points[-1]}\tResiduum: {self.distances[-1]}")
            self.good_res_plot.append(self.distances[-1])

        # add points that are better than a given threshold just for plotting
        if self.distances[-1] <= 1e-5:
            self.good_points_plot.append(action)

        if discrete:
            self.state = action

            if residuum < self.best_residuum:
                self.best_residuum = residuum

            self.min_residuum = min(self.min_residuum, residuum)
            self.max_residuum = max(self.max_residuum, residuum)

            # normalie residuum
            if self.max_residuum - self.min_residuum != 0:
                normalized_residuum = (residuum - self.min_residuum) / (self.max_residuum - self.min_residuum)
            else:
                normalized_residuum = 0

            residuum = normalized_residuum

            # print(f"Normalized Residuum: {normalized_residuum}")

            # reward = np.exp(-normalized_residuum * 1000)
            # reward = 1 - residuum
            # reward = np.log(1 / (normalized_residuum + 1e-8))
            reward = self.adaptive_reward(normalized_residuum, self.min_residuum, self.max_residuum, reward_type='logarithmic',
                                          alpha_init=1.0, beta_init=1.0, gamma=0.1, state=self.state)

            if isinstance(reward, int) or isinstance(reward, float):
                self.rewards.append(reward)
                self.reward_action_map.append((action, reward))
            else:
                self.rewards.append(reward[0])
                self.reward_action_map.append((action, reward[0]))

            if residuum > self.good_points_thrs:
                reward -= residuum

            # print(f"Reward: {reward}")

            if residuum <= self.good_points_thrs:
                self.good_points.append(action)

            # Reward if action is better than last action
            if len(self.distances) > 1 and residuum < self.distances[-2]:
                reward += residuum
                self.best_action_so_far = action

            # penalty if action is worse than last action
            if len(self.best_actions) > 1 and residuum > self.best_distances[-2]:
                reward -= residuum

            self.time_residuum.append((time.time() - self.start_time, self.best_residuum))
            # self.time_action.append((time.time() - self.start_time, action))

            done = False
            truncated = False
            # reward = np.clip(reward, 0, 1)

            epsilon = 1e-8
            max_residuum = 100.0  # Determine the maximum possible residuum value based on your problem

            max_reward = 1 / epsilon
            min_reward = 1 / (max_residuum + epsilon)

            # calculate improvement rate
            if len(self.best_distances) > 1:
                improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
                self.improvement_rate_history.append(improvement_rate)

            if residuum <= self.good_points_thrs:  # if distance is less than reset
                print("Reset")
                done = True
                reward = 1 + (self.good_points_thrs - residuum) / self.good_points_thrs

                if len(self.best_distances) > 1:
                    improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
                    if improvement_rate > 0.25:
                        self.good_points_thrs *= 0.6  # Aggressive decrease for significant improvement
                    else:
                        self.good_points_thrs *= 0.98  # Gradual decrease for slow improvement
                else:
                    self.good_points_thrs *= 0.9

                # done = True
                # reward = 1 + (self.good_points_thrs - residuum) / self.good_points_thrs
                #
                # if len(self.best_distances) > 1:
                #     # Calculate the current improvement rate
                #     current_improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / \
                #                                self.best_distances[-2]
                #
                #     # Update the improvement rate history
                #     self.improvement_rate_history.append(current_improvement_rate)
                #     if len(self.improvement_rate_history) > self.improvement_rate_window_size:
                #         self.improvement_rate_history.pop(0)
                #
                #     # Calculate the adaptive threshold
                #     sorted_improvement_rates = sorted(self.improvement_rate_history, reverse=True)
                #     adaptive_threshold_index = int(0.8 * len(sorted_improvement_rates))  # Adjust the quantile as needed
                #     adaptive_threshold = sorted_improvement_rates[adaptive_threshold_index]
                #
                #     if current_improvement_rate > adaptive_threshold:
                #         self.good_points_thrs *= 0.8  # Aggressive decrease for significant improvement
                #     else:
                #         self.good_points_thrs *= 0.95  # Gradual decrease for slow improvement
                # else:
                #     self.good_points_thrs *= 0.9



            #     done = True
            #     reward = 1 + (self.good_points_thrs - residuum) / self.good_points_thrs
            #     self.good_points_thrs *= max(0.9, self.good_points_thrs * 0.99)
            #     self.consecutive_no_improvement = 0
            # else:
            #     self.consecutive_no_improvement += 1
            #
            #     if self.consecutive_no_improvement >= 100:
            #         done = True
            #         reward = 0

            # reward = 1 / (residuum + epsilon)
            # reward = (reward - min_reward) / (max_reward - min_reward)
            # self.good_points_thrs *= max(0.9, self.good_points_thrs * 0.99)

            # done = True
            # reward = -np.log(residuum + epsilon)
            # reward = (reward - min_reward) / (max_reward - min_reward)
            # self.good_points_thrs *= max(0.9, self.good_points_thrs * 0.99)
            else:
                reward = -np.log(residuum + epsilon)
                reward = (reward - min_reward) / (max_reward - min_reward)

            # else:
            #     reward = 1 / (residuum + epsilon)
            #     reward = (reward - min_reward) / (max_reward - min_reward)

            # self.best_action.append(action)
            self.last_obs = self.state

            # reward = np.clip(reward, 0, 1)

            return self.state, reward, done, truncated, {}

    def reset(self, seed=None):
        """
        Reset the state of the environment and return an initial observation.
        :param seed: Seed must be set
        :return: state, {}
        """
        discrete = True

        # print("Reset")

        self.state = np.array(random.uniform(self.x_min, self.x_max))

        # if len(self.good_points) > 0:
        #    self.action_space = gym.spaces.Box(low=min(self.good_points), high=max(self.good_points))

        # self.best_action = []
        self.last_obs = self.state

        return self.state, {}


if __name__ == '__main__':
    # --- plot discrete nse ---
    # equations = discrete_nse(-10.0, 8.0, -10.0, 10.0)
    # plot_discrete_nse(equations, -10.0, 8.0, -10.0, 10.0)

    # init environment
    dimension = 2
    env = CustomEnv(dimension=dimension)
    logger.info("Environment created")

    # env._plot_res(dimension, plot_contour=True)

    # find minimum of residuum

    # check environment
    # check_env(env)

    # create model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, ent_coef=0.15)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.8 * np.ones(n_actions))
    # model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
    #             buffer_size=1000)

    # model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
    #             learning_starts=1000)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.2)
    logger.info("Model created")

    # train model
    log = False
    if log:
        actions = []
        callback = SaveActionsCallback(1, actions)
        logger.info("Start training")
        model.learn(total_timesteps=int(1e5), progress_bar=False, tb_log_name="PPO_NSE")
    else:
        actions = []
        callback = SaveActionsCallback(1, actions)
        logger.info("Start training")
        start_training_time = time.time()
        model.learn(total_timesteps=int(11e4), progress_bar=False)
        print(f"Training time: {round(time.time() - start_training_time, 3)}s")

    _, distances = env.best_actions, env.distances
    good_points, good_points_plot = env.good_points, env.good_points_plot
    good_res_plot = env.good_res_plot
    best_residuen = env.best_distances
    rewards = env.rewards

    # calculate best point away from global minimum
    global_optimum = (1,1)
    point_distance_map = dict()
    for point in good_points:
        distance = np.linalg.norm(np.array(point) - np.array(global_optimum))
        # make ndarray to tuple
        point_tuple = tuple(point)
        point_distance_map[point_tuple] = distance

    best_point = min(point_distance_map, key=point_distance_map.get)

    print(f"Best point away from global minimum: {best_point}")

    # plot time_residuum and time_action in subplots
    time_residuum_map = env.time_residuum
    # time_action_map = env.time_action

    fig, axs = plt.subplots(2, 1, figsize=(15, 9))
    times_residuum = [time for time, residuum in time_residuum_map]
    residuums = [residuum for time, residuum in time_residuum_map]
    # times_action = [time for time, action in time_action_map]
    # actions = [action for time, action in time_action_map]

    axs[0].plot(times_residuum, residuums)
    axs[0].set_title("Time Residuum")
    axs[0].set_xlabel("Time (in s)")
    axs[0].set_ylabel("Residuum")
    axs[0].grid()

    # scale logarithmic
    axs[0].set_yscale('log')

    # axs[1].plot(times_action, actions)
    # axs[1].set_title("Time Action")
    # axs[1].set_xlabel("Time")
    # axs[1].set_ylabel("Action")

    plt.show()

    print(good_points)
    print(f"Number of good points: {len(good_points)}")
    # print(f"Good points: {good_points}")

    print(f"Residuum Average: {np.mean(distances)}")

    x_min = -3.0
    x_max = 3.0
    y_min = -3.0
    y_max = 3.0

    # plot improvement rate history
    plt.figure(figsize=(8, 6))
    plt.plot(env.improvement_rate_history)
    plt.xlabel('Step')
    plt.ylabel('Improvement Rate')
    plt.title('Improvement Rate History')
    plt.grid()
    plt.show()

    nse = env.nse()
    plot_nse(dimension=2, equations=nse, x_array=np.linspace(x_min, x_max, 100), points=good_points_plot,
             contour_plot=True)

    # plot best residuen
    best_distances = env.best_distances

    # plot reward action mapping
    # actions = [action for action, reward in env.reward_action_map]
    # rewards = [reward for action, reward in env.reward_action_map]
    # plt.scatter(actions, rewards, s=1.2, label="reward = log(1 / residuum)")
    # # real_solutions = [-0.69983978673645688906, 0.0, 2.4936955491125650267]
    # # for real_sol in real_solutions:
    # #     plt.axvline(x=real_sol, color='red', label='Real Solution')
    # plt.xlabel("Action")
    # plt.ylabel("Reward")
    # plt.xlim(env.x_min, env.x_max)
    # plt.legend()
    # plt.show()

    # save model
    # model.save("ddpg_nse")

    # tensorboard befehl: tensorboard --logdir ./tmp/
