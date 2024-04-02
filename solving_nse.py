import gymnasium as gym
import random
import math
import numpy as np
import pandas as pd
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
        fig, axs = plt.subplots(1, 2, figsize=(9, 7))

        x_values = [x[0] for x in points]
        eq1 = equations[0]
        y_values = [eq1(x) for x in points]

        for eq in equations:
            y = eq(x_array)
            axs[0].plot(x_array, y)

        if points is not None:
            axs[0].scatter(x_values, y_values, color='red', label='Points')
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


def rosenbrock_gradient(x):
    df_dx = -40 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    df_dy = 20 * (x[1] - x[0] ** 2)
    return np.array([df_dx, df_dy])


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
                                           dtype=np.float64)

        self.observation_space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(dimension,))

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
        self.all_actions = []
        self.all_epoch_time_action_residuals = []

        self.all_residuals = list()

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
        if self.dimension == 1:
            eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
            eq2 = lambda x: np.sin(2 * x)
            return np.array([eq1, eq2])

        elif self.dimension == 2:
            # rosenbrock function
            rosenbrock_eq1 = lambda x: 10 * (x[1] - x[0] ** 2)
            rosenbrock_eq2 = lambda x: 1 - x[0]
            return [rosenbrock_eq1, rosenbrock_eq2]

        else:
            n = 10
            list_of_eqs = []

            # make own function for every k
            for k in range(1, n):
                # equation = (x_k + sum(x_i * x_{i+k} for i=1 to n-1))*x_n
                def eq(x, k=k):  # use default argument to capture current value of k
                    summe = sum(x[i] * x[(i + k)] for i in range(0, n - k - 1))
                    return (x[k - 1] + summe) * x[n - 1]

                list_of_eqs.append(eq)

            eq2 = lambda x: sum(x[l] for l in range(0, n))
            list_of_eqs.append(eq2)

            return list_of_eqs


    def get_distance_discrete(self, point):
        """
        Calculate the residuum for the given point.
        :param point:
        :param nse:
        :return:
        """
        scaling = "exponential"
        equations = self.nse()

        # Calculate the function values for all equations
        values = np.array([eq(point) for eq in equations])

        # calculate normal residuum
        normal_res = sum((values[i] - values[j]) ** 2 for i in range(len(values)) for j in range(i + 1, len(values)))
        self.all_residuals.append(normal_res)

        if scaling == "minmax":
            if len(self.all_residuals) > 1:
                min_values = np.min(self.all_residuals, axis=0)
                max_values = np.max(self.all_residuals, axis=0)
                scaled_values = (values - min_values) / (max_values - min_values)
                res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                          range(i + 1, len(scaled_values)))
                return res, normal_res
            else:
                return normal_res, normal_res

        elif scaling == "logarithmic":
            scaled_values = np.log1p(np.abs(values)) / np.log(10)
            res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                      range(i + 1, len(scaled_values)))
            return res, normal_res

        elif scaling == "exponential":
            scaled_values = np.exp(-np.abs(values) * 1000)
            res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                      range(i + 1, len(scaled_values)))
            return res, normal_res

        elif scaling == "zscore":
            if len(self.all_residuals) > 1:
                mean_values = np.mean(self.all_residuals, axis=0)
                std_values = np.std(self.all_residuals, axis=0)
                scaled_values = (values - mean_values) / std_values
                res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                          range(i + 1, len(scaled_values)))
                return res, normal_res
            else:
                return normal_res, normal_res

        elif scaling == "normal":
            return normal_res, normal_res

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
            x = np.linspace(-5, 5, num_points)
            y = np.linspace(-5, 5, num_points)
            X, Y = np.meshgrid(x, y)
            points = np.stack((X, Y), axis=-1)

            residuals = np.array([self.get_distance_discrete(point)[1] for point in points.reshape(-1, 2)])
            Z = residuals.reshape(X.shape)

            if plot_contour:
                plt.figure(figsize=(8, 6))
                # contourf plot
                plt.contourf(X, Y, Z, cmap='viridis', levels=100)

                # make point at (1,1)
                plt.scatter(1, 1, color='red', label='Minimum')

                # plot good points
                for i, good_point in enumerate(self.good_points_plot):
                    if i == 0:
                        plt.scatter(good_point[0], good_point[1], color='green', label='Result')
                    else:
                        plt.scatter(good_point[0], good_point[1], color='green')


                plt.colorbar(label='Residuum')
                plt.legend()
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
            x = state
            gradient = rosenbrock_gradient(x)
            gradient_term = -gamma * np.linalg.norm(gradient)
        else:
            gradient_term = 0

        reward = reward_term + normalization_term + gradient_term

        return reward

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action to take.
        :return: state, reward, done, info
        """

        discrete = True
        self.actions.append(action)
        residuum, normal_residuum = self.get_distance_discrete(action)
        self.distances.append(normal_residuum)
        self.all_actions.append(action)

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
        if self.distances[-1] <= 1e-3:
            self.good_points_plot.append(action)

        if discrete:
            self.state = action

            if normal_residuum < self.best_residuum:
                self.best_residuum = normal_residuum

            self.min_residuum = min(self.min_residuum, normal_residuum)
            self.max_residuum = max(self.max_residuum, normal_residuum)

            # normalize residuum
            # if self.max_residuum - self.min_residuum != 0:
            #     normalized_residuum = (residuum - self.min_residuum) / (self.max_residuum - self.min_residuum)
            # else:
            #     normalized_residuum = 0
            #
            # residuum = normalized_residuum

            # print(f"Normalized Residuum: {normalized_residuum}")

            # reward = np.exp(-normalized_residuum * 1000)
            # reward = 1 - residuum
            # reward = np.log(1 / (normalized_residuum + 1e-8))

            # reward = self.adaptive_reward(normalized_residuum, self.min_residuum, self.max_residuum,
            #                               reward_type='logarithmic',
            #                               alpha_init=0.99, beta_init=0.99, gamma=0.1, state=self.state)

            # reward = self.logarithmic_reward(residuum, self.min_residuum, self.max_residuum, alpha=1.0, beta=1.0)

            reward = -residuum

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
            # if len(self.distances) > 1 and residuum < self.distances[-2]:
            #     reward += residuum
            #     self.best_action_so_far = action

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
                # print("Reset")
                done = True
                # reward = 1 + (self.good_points_thrs - residuum) / self.good_points_thrs

                if len(self.best_distances) > 1:
                    improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
                    if improvement_rate > 0.25:
                        self.good_points_thrs *= 0.1  # Aggressive decrease for significant improvement
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
            # self.last_obs = self.state

            # reward = np.clip(reward, 0, 1)
            self.all_epoch_time_action_residuals.append((time.time() - self.start_time, action, normal_residuum))

            return self.state, reward, done, truncated, {}

    def reset(self, seed=None):
        """
        Reset the state of the environment and return an initial observation.
        :param seed: Seed must be set
        :return: state, {}
        """

        # update based on dimension
        self.state = np.random.uniform(low=self.x_min, high=self.x_max, size=self.dimension)

        return self.state, {}


def normal_train_eval(epochs: float, dimension: int, model: str):
    import time

    # Create environment
    env = CustomEnv(dimension=dimension)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.75 * np.ones(n_actions))

    # Create model
    if model == "DDPG":
        model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
                     buffer_size=1000)
    elif model == "SAC":
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
                    learning_starts=1000)
    elif model == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.15)

    logger.info("Model created")

    # Train model
    logger.info(f"Start training. Epochs: {epochs}; Model: {model}; Dimension: {dimension}")
    start_training_time = time.time()
    model.learn(total_timesteps=int(epochs), progress_bar=False)
    print("---" * 100)
    print(f"\nTraining time: {round(time.time() - start_training_time, 3)}s")
    print("===" * 100)

    if dimension == 1:
        # plot residuum
        plot_nse(env.nse(), np.linspace(-10, 10, 10000), dimension=dimension, contour_plot=False, points=env.good_points_plot,
                 points_x=None, distances=env.best_distances)

    if dimension == 2:
        print(f"Len of good points plot: {len(env.good_points_plot)}")
        plot_nse(env.nse(), np.linspace(-5, 5, 100), dimension=dimension, contour_plot=True, points=env.good_points_plot,
                    points_x=None, distances=env.best_distances)

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

    good_points = env.good_points
    distances = env.distances

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


def benchmark(epochs: list):
    df = pd.DataFrame(columns=['Epochs', 'Time', 'Action', 'Residuum'])
    for epoch in epochs:
        dimension = 2
        env = CustomEnv(dimension=dimension)

        # action noise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.75 * np.ones(n_actions))
        # model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
        #             buffer_size=1000)

        # model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
        #             learning_starts=1000)
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.15)
        logger.info("Model created")

        actions = []
        callback = SaveActionsCallback(1, actions)
        logger.info("Start training")
        start_training_time = time.time()
        model.learn(total_timesteps=int(epoch), progress_bar=False)
        print("---" * 100)
        print(f"\nTraining time: {round(time.time() - start_training_time, 3)}s")
        print("===" * 100)

        df = df.append({'Epochs': epoch, 'Time': round(time.time() - start_training_time, 3),
                        'Action': actions, 'Residuum': env.best_residuum}, ignore_index=True)
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"benchmark_{date}.csv", index=False)


if __name__ == '__main__':
    epochs = 6e3
    dimension = 2
    model = "PPO"
    normal_train_eval(epochs, dimension, model)

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
