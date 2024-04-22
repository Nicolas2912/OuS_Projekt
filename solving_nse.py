import gymnasium as gym
import random
import math
import numpy as np
import pandas as pd
import scipy.special as sp

import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, TD3, SAC
import os
# import numdifftools as nd
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
import structlog
from tqdm import tqdm

import torch

# np.random.seed(41)

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

        eqs_names = ["x^5 - 3x^4 + x^3 + 0.5x^2 - sin(2x)"]

        for idx, eq in enumerate(equations):
            y = eq(x_array)
            axs[0].plot(x_array, y, label=f"{eqs_names[idx]}")

        if points is not None:
            axs[0].scatter(x_values, y_values, color='red', label='Found solutions')
        if points_x is not None:
            for x_value in points_x:
                axs[0].axvline(x=x_value, color='red')

        if distances is not None:
            axs[1].plot(distances, marker='o')

        axs[0].set_title("NSE 1 with solutions")
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        # x axis at 0
        axs[0].axhline(0, color='black', lw=0.5)

        axs[0].legend()
        axs[0].grid()
        axs[0].set_ylim(-3.0, 3.0)
        axs[0].set_xlim(-3.0, 3.0)

        axs[1].set_title("Best Residuen")
        axs[1].set_yscale('log')
        axs[1].grid()

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
            if points is not None:

                for i, point in enumerate(points):
                    # plot points but just one label
                    if i == 0:
                        plt.scatter(point[0], point[1], color='red', label='Found Solutions')
                    else:
                        plt.scatter(point[0], point[1], color='red')

            # plot real minimum at (1,1) in blue
            plt.scatter(1, 1, color='blue', label='Global Minimum')

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
    def __init__(self, dimension, improvement_rate_thr=0.95, scaling="normal"):
        super(CustomEnv, self).__init__()
        self.x_min = -10.0
        self.x_max = 10.0
        self.y_min = -5.0
        self.y_max = 5.0

        self.global_min, self.global_max = -5.0, 5.0
        self.local_min, self.local_max = self.global_min, self.global_max
        self.narrowing_factor = 0.9
        self.expansion_factor = 1.1

        self.dimension = dimension
        self.low_bounds = -5.0 * np.ones(dimension)
        self.high_bounds = 5.0 * np.ones(dimension)
        self.action_space = gym.spaces.Box(low=self.low_bounds,
                                           high=self.high_bounds,
                                           dtype=np.float64)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dimension,), dtype=np.float64)

        self.state = np.array(
            [random.uniform(self.x_min, self.x_max),
             random.uniform(self.y_min, self.y_max)])  # self.state = point in space

        self.improvement_rate_thr = improvement_rate_thr
        self.scaling = scaling

        self.best_actions = []
        self.last_obs = None
        self.actions = []
        self.distances = []
        self.best_distances = []
        self.good_points = []
        self.good_points_plot = []
        self.good_res_plot = []
        self.best_action_so_far = None
        self.good_points_thrs = 0.01
        self.good_points_thrs_list = []
        self.episode = 0
        self.rewards = []
        self.reward_action_map = []
        self.all_actions = []
        self.all_epoch_time_action_residuals = []
        self.global_optimum = None

        self.all_residuals = list()

        self.consecutive_no_improvement = 0
        self.total_steps = 0

        self.improvement_rate_window_size = 10
        self.improvement_rate_history = []

        self.action_precision = dict()
        self.time_residuum = list()
        self.time_reward = list()
        self.time_action = list()

        self.min_residuum = float('inf')
        self.max_residuum = float('-inf')

        self.best_residuum = float('inf')
        self.last_residuum = float('inf')
        self.low_residuum_threshold = 1e-6
        self.small_step_reward_factor = 0.25

        self.start_time = time.time()

    def calculate_derivative(self, action):
        """
        derivative of each function of nse 1 (dimension 1)
        :param action: action
        :return: array of derivatives
        """
        d_eq1 = lambda x: 5 * x ** 4 - 12 * x ** 3 + 3 * x ** 2 + x
        d_eq2 = lambda x: 2 * np.cos(2 * x)

        return np.array([d_eq1(action), d_eq2(action)])

    def calculate_derivative_rosenbrock(self, action):
        """
        derivative of each function of nse 1 (dimension 1)
        :param action: action
        :return: array of derivatives
        """
        # Partial derivative with respect to x[0]
        dx0 = -20 * action[0] - 1

        # Partial derivative with respect to x[1]
        dx1 = 10

        return np.array([dx0, dx1])

    def nse(self):
        if self.dimension == 1:
            eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
            eq2 = lambda x: np.sin(2 * x)
            eq_r = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2 - np.sin(2 * x)
            self.global_optimum = [-0.6998397867, 0.0, 2.4936955491]
            return np.array([eq_r])

        elif self.dimension == 2:
            # rosenbrock function
            rosenbrock_eq1 = lambda x: 10 * (x[1] - x[0] ** 2)
            rosenbrock_eq2 = lambda x: 1 - x[0]
            self.global_optimum = [1, 1]
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
            self.global_optimum = [-1.10488041e+00, 4.60954938e-01, 9.38515375e-01, 3.96169982e-01,
                                   -1.80518000e-01, -4.59316409e-01, -6.19859873e-01, -2.92935873e-01,
                                   -1.38129732e-01, 1.54675389e-17]  # for n=10

            return list_of_eqs

    def get_distance_discrete(self, point):
        """
        Calculate the residuum for the given point.
        :param point:
        :param nse:
        :return:
        """

        equations = self.nse()

        # Calculate the function values for all equations
        values = np.array([eq(point) for eq in equations])
        residuum = np.sqrt(sum(values ** 2))

        # calculate normal residuum
        normal_res = sum((values[i] - values[j]) ** 2 for i in range(len(values)) for j in range(i + 1, len(values)))
        self.all_residuals.append(residuum)

        if self.scaling == "minmax":
            if len(self.all_residuals) > 1:
                min_values = np.min(self.all_residuals, axis=0)
                max_values = np.max(self.all_residuals, axis=0)
                scaled_residuum = (residuum - min_values) / (max_values - min_values)
                return scaled_residuum, residuum
            else:
                return residuum, residuum

        elif self.scaling == "logarithmic":
            scaled_residuum = np.log(residuum + 1e-8)
            return scaled_residuum, -scaled_residuum

        elif self.scaling == "exponential":
            scaled_residuum = np.exp(-0.5 * (residuum) ** 2)
            return scaled_residuum, residuum

        elif self.scaling == "gradient":
            # scale values with square of gradient
            if self.dimension == 1:
                gradient = self.calculate_derivative(point)
                scaled_values = values / (1 + np.abs(gradient) * normal_res)
                res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                          range(i + 1, len(scaled_values)))
                return res, normal_res
            elif self.dimension == 2:
                gradient = rosenbrock_gradient(point)
                scaled_values = values / (1 + np.abs(gradient))
                res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                          range(i + 1, len(scaled_values)))
                return res, normal_res

        elif self.scaling == "sigmoid":
            # sigmoid scaling
            scaled_residuum = 1 / (1 + np.exp(-residuum))
            return scaled_residuum, residuum

        elif self.scaling == "zscore":
            if len(self.all_residuals) > 1:
                mean_values = np.mean(self.all_residuals, axis=0)
                std_values = np.std(self.all_residuals, axis=0)
                scaled_values = (values - mean_values) / std_values
                res = sum((scaled_values[i] - scaled_values[j]) ** 2 for i in range(len(scaled_values)) for j in
                          range(i + 1, len(scaled_values)))
                return res, normal_res
            else:
                return normal_res, normal_res

        elif self.scaling == "normal":
            inv = 1 / (residuum + 1)
            # gaussian = np.exp(-((residuum**2) / (1)))
            # cosine = np.cos(min(residuum, np.pi / 2))
            # log = -np.log(residuum + 1)**2+1
            # square_root = 1/(np.sqrt(residuum + 1e-8))
            tanh = -np.tanh(residuum ** 2) + 1
            test = -1 / (residuum + 1) * np.tanh(residuum ** 2) + 1 * np.exp(-residuum ** 2)
            # erf = sp.erf(1/(residuum + 1))
            return inv, residuum

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

    def reward_penalty_simple(self, residuum):
        """
        Reward function that penalizes high residuum values and rewards low residuum values.
        :param residuum:
        :param reward:
        :return:
        """
        done = False
        truncated = False
        reward = residuum
        if len(self.rewards) > 100:
            self.rewards.pop(0)

        if residuum < self.good_points_thrs:
            self.good_points_thrs *= 0.8
            done = True
        else:
            reward -= 0.5

        return reward, done, truncated

    def dynamic_reward_penalty(self, action, residuum, reward):
        dynamic_reward = 0.0
        dynamic_penalty = 0.0
        reward_scaling_factor = 1.0
        penalty_scaling_factor = 1.0

        # penalty if residuum is greater than threshold
        if residuum > self.good_points_thrs:
            dynamic_reward += residuum * reward_scaling_factor
        else:
            dynamic_penalty += residuum * penalty_scaling_factor

        # print(f"Reward: {reward}")
        if residuum <= self.good_points_thrs:
            self.good_points.append(action)

        # Reward if action is better than last action
        if len(self.distances) > 1 and residuum < self.distances[-2]:
            residuum_difference_reward = self.distances[-2] - residuum
            dynamic_reward += residuum_difference_reward

            self.best_action_so_far = action

        # penalty if action is worse than last action
        if len(self.best_actions) > 1 and residuum > self.best_distances[-2]:
            residuum_difference_penalty = residuum - self.distances[-2]
            dynamic_penalty += residuum_difference_penalty

        reward += dynamic_reward
        reward -= dynamic_penalty

        self.time_residuum.append((time.time() - self.start_time, self.best_residuum))
        # self.time_action.append((time.time() - self.start_time, action))

        done = False
        truncated = False

        improvement_rate = 0.0

        if len(self.best_distances) > 1:
            improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
            self.improvement_rate_history.append(improvement_rate)

            # Adaptive threshold
            self.good_points_thrs = max(0.9, self.good_points_thrs * (1 - improvement_rate))

            # Variable consecutive no improvement limit
            no_improvement_limit = max(50, 100 * improvement_rate)

        else:
            no_improvement_limit = 50

        if residuum <= self.good_points_thrs:  # if distance is less than reset
            done = True
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

            if self.consecutive_no_improvement >= no_improvement_limit:
                done = True
                reward = 0

        # Reward shaping
        if improvement_rate > 0.01:  # Threshold can be adjusted
            reward += 1  # Bonus can be adjusted

        return reward, done, truncated

    def dynamic_reward_penalty_ss(self, action, residuum, reward):
        dynamic_reward = 0.0
        dynamic_penalty = 0.0
        reward_scaling_factor = 1.0
        penalty_scaling_factor = 1.0

        # penalty if residuum is greater than threshold
        if residuum > self.good_points_thrs:
            dynamic_reward += residuum * reward_scaling_factor
        else:
            dynamic_penalty += residuum * penalty_scaling_factor

        # print(f"Reward: {reward}")
        if residuum <= self.good_points_thrs:
            self.good_points.append(action)

        # Reward if action is better than last action
        if len(self.distances) > 1 and residuum < self.distances[-2]:
            residuum_difference_reward = self.distances[-2] - residuum
            dynamic_reward += residuum_difference_reward * (self.distances[-2] - residuum)

            self.best_action_so_far = action

        # penalty if action is worse than last action
        if len(self.best_actions) > 1 and residuum > self.best_distances[-2]:
            residuum_difference_penalty = residuum - self.distances[-2]
            dynamic_penalty += residuum_difference_penalty * (residuum - self.distances[-2])

        reward += dynamic_reward
        reward -= dynamic_penalty

        self.time_residuum.append((time.time() - self.start_time, self.best_residuum))
        # self.time_action.append((time.time() - self.start_time, action))

        done = False
        truncated = False

        improvement_rate = 0.0

        if len(self.best_distances) > 1:
            improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
            self.improvement_rate_history.append(improvement_rate)

            # Adaptive threshold
            self.good_points_thrs = max(0.9, self.good_points_thrs * (1 - improvement_rate))

            # Variable consecutive no improvement limit
            no_improvement_limit = max(50, 100 * improvement_rate)

        else:
            no_improvement_limit = 50

        if residuum <= self.good_points_thrs:  # if distance is less than reset
            done = True
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

            if self.consecutive_no_improvement >= no_improvement_limit:
                done = True
                reward = 0

        # Reward shaping
        if improvement_rate > 0.01:  # Threshold can be adjusted
            reward += 1  # Bonus can be adjusted

        # Encourage smaller steps when residuum is low
        if residuum < self.low_residuum_threshold:
            action_magnitude = np.linalg.norm(action)
            small_step_reward = self.small_step_reward_factor * np.exp(-action_magnitude ** 2)
            reward += small_step_reward

        return reward, done, truncated

    def dynamic_reward_penalty_gradient_scale(self, action, residuum, normal_residuum, reward):
        dynamic_reward = 0.0
        dynamic_penalty = 0.0
        reward_scaling_factor = 1.0
        penalty_scaling_factor = 1.0

        # Calculate the gradient at the given action
        gradient = self.calculate_derivative(action)

        # Scale the residuum using the gradient information
        scaled_residuum = residuum / (1 + np.abs(gradient) * normal_residuum)

        # squared sum of scaled residuum
        scaled_residuum = sum(scaled_residuum[i] - scaled_residuum[j] for i in range(len(scaled_residuum)) for j in
                              range(i + 1, len(scaled_residuum)))

        scaled_residuum = sum(scaled_residuum)

        # Penalty if scaled residuum is greater than threshold
        if scaled_residuum > self.good_points_thrs:
            dynamic_penalty += scaled_residuum * penalty_scaling_factor
        else:
            dynamic_reward += scaled_residuum * reward_scaling_factor

        if scaled_residuum <= self.good_points_thrs:
            self.good_points.append(action)

        # Reward if action is better than last action
        if len(self.distances) > 1 and scaled_residuum < self.distances[-2]:
            residuum_difference_reward = self.distances[-2] - scaled_residuum
            dynamic_reward += residuum_difference_reward * (self.distances[-2] - scaled_residuum)

            self.best_action_so_far = action

        # Penalty if action is worse than last action
        if len(self.best_actions) > 1 and scaled_residuum > self.best_distances[-2]:
            residuum_difference_penalty = scaled_residuum - self.distances[-2]
            dynamic_penalty += residuum_difference_penalty * (scaled_residuum - self.distances[-2])

        reward = -scaled_residuum
        reward += dynamic_reward
        reward -= dynamic_penalty

        self.time_residuum.append((time.time() - self.start_time, self.best_residuum))

        done = False
        truncated = False

        improvement_rate = 0.0

        if len(self.best_distances) > 1:
            improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
            self.improvement_rate_history.append(improvement_rate)

            # Adaptive threshold
            self.good_points_thrs = max(0.9, self.good_points_thrs * (1 - improvement_rate))

            # Variable consecutive no improvement limit
            no_improvement_limit = max(50, 100 * improvement_rate)

        else:
            no_improvement_limit = 50

        if scaled_residuum <= self.good_points_thrs:
            done = True
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

            if self.consecutive_no_improvement >= no_improvement_limit:
                done = True
                reward = 0

        # Reward shaping
        if improvement_rate > 0.01:  # Threshold can be adjusted
            reward += 1  # Bonus can be adjusted

        return reward, done, truncated

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action to take.
        :return: state, reward, done, info
        """
        done = False
        truncated = False

        self.actions.append(action)
        residuum, normal_residuum = self.get_distance_discrete(action)
        self.distances.append(normal_residuum)
        self.action_precision[tuple(action)] = normal_residuum
        self.time_action.append((time.time() - self.start_time, action))
        self.time_residuum.append((time.time() - self.start_time, normal_residuum))

        improvement = normal_residuum - self.last_residuum
        self.last_residuum = normal_residuum

        # just print better action
        if self.distances[-1] <= min(self.distances):
            print("Best action:", self.actions[-1], "\t", "Residuum:", self.distances[-1], "\t", "Episode:",
                  self.episode)
            # print()
            self.best_actions.append(action)
            self.best_distances.append(self.distances[-1])
            self.best_residuum = residuum
            self.local_min = max(self.global_min, action - self.narrowing_factor)
            self.local_max = min(self.global_max, action + self.narrowing_factor)

            # print(f"Action space: {self.local_min} - {self.local_max}")

        # self.action_space = gym.spaces.Box(low=np.array([self.local_min]), high=np.array([self.local_max]), dtype=np.float64)

        if self.distances[-1] <= self.good_points_thrs and len(self.good_points) > 0:
            # print(f"Good point: {self.good_points[-1]}\tResiduum: {self.distances[-1]}")
            self.good_res_plot.append(self.distances[-1])

        # add points that are better than a given threshold just for plotting
        if self.distances[-1] <= 0.001:
            self.good_points_plot.append(action)
            # print(f"Good point: {action}\tResiduum: {self.distances[-1]}")

        self.state = action

        # if normal_residuum < self.best_residuum:
        #     self.best_residuum = normal_residuum

        self.min_residuum = min(self.min_residuum, normal_residuum)
        self.max_residuum = max(self.max_residuum, normal_residuum)

        # reward = 1/(residuum + 1)
        reward = residuum

        # --- standard way ---

        if normal_residuum < self.best_residuum:
            self.best_residuum = normal_residuum
            self.best_action_so_far = action
            reward += 1.0
        else:
            reward -= 0.1

        if improvement < 0:
            reward += 0.1
        else:
            reward -= 0.1

        if normal_residuum < self.good_points_thrs:
            reward += 0.05  # Simple reward for being below threshold
            self.good_points_thrs *= 0.95  # Adjust threshold
            self.consecutive_no_improvement = 0
        else:
            reward -= 0.01
            self.consecutive_no_improvement += 1
            if self.consecutive_no_improvement >= 50:
                done = True
                reward -= 0.02

        # --- standard way ---

        self.time_reward.append((time.time() - self.start_time, reward))
        # print(f"Residuum: {residuum}\tReward: {reward}")
        # if residuum == 0:
        #     print(f"Action: {action} is the global minimum.")

        if isinstance(reward, int) or isinstance(reward, float):
            self.rewards.append(reward)
            self.reward_action_map.append((action, reward))
        else:
            self.rewards.append(reward[0])
            self.reward_action_map.append((action, reward[0]))

        # reward shaping
        # reward, done, truncated = self.dynamic_reward_penalty_gradient_scale(action, residuum, normal_residuum, reward)
        # reward, done, truncated = self.reward_penalty_simple(residuum)
        # reward, done, truncated = self.dynamic_reward_penalty(action, residuum, reward)
        self.good_points_thrs_list.append(self.good_points_thrs)

        if done:
            self.episode += 1

        # print every 100 episodes
        # if self.episode % 100 == 0:
        #     print(f"Episode: {self.episode}\tResiduum: {normal_residuum}\tBest Residuum: {self.best_residuum}")

        self.all_epoch_time_action_residuals.append((time.time() - self.start_time, action, normal_residuum))

        return self.state, reward, done, truncated, {}

    def reset(self, seed=None):
        """
        Reset the state of the environment and return an initial observation.
        :param seed: Seed must be set
        :return: state, {}
        """

        if self.dimension == 2:
            # Initialize variables for tracking improvement
            if not hasattr(self, 'no_improvement_duration'):
                self.no_improvement_duration = 0
                self.last_best_residuum = float('inf')
                self.improvement_threshold = 0.01  # Initial threshold for significant improvement

            exploration_range = max(0.1, 10 / (self.episode + 1))
            step_size_factor = 0.01  # Adjust the factor as needed

            if len(self.best_actions) > 0:
                # Select the best point from the best points
                best_point = self.best_actions[
                    np.argmin([self.get_distance_discrete(point)[0] for point in self.best_actions])]
                best_residuum = self.get_distance_discrete(best_point)[0]

                # Update improvement tracking variables
                if best_residuum < self.last_best_residuum:
                    improvement = (self.last_best_residuum - best_residuum) / self.last_best_residuum
                    if improvement > self.improvement_threshold:
                        self.no_improvement_duration = 0
                        self.improvement_threshold = max(0.001,
                                                         improvement * 0.5)  # Adjust the threshold based on the improvement
                    else:
                        self.no_improvement_duration += 1
                    self.last_best_residuum = best_residuum
                else:
                    self.no_improvement_duration += 1

                # Calculate the gradient at the best point
                gradient = self.calculate_derivative_rosenbrock(best_point)

                # Check if the gradient is not zero
                if np.linalg.norm(gradient) > 1e-8:
                    # Normalize the gradient vector
                    gradient /= np.linalg.norm(gradient)

                    # Calculate the step size based on the exploration range, step size factor, and no improvement duration
                    step_size = exploration_range * step_size_factor / (1 + 0.1 * self.no_improvement_duration)

                    # Generate a new point by taking a step in the opposite direction of the gradient
                    new_point = best_point - step_size * gradient

                    # Ensure the new point stays within the valid range
                    new_point = np.clip(new_point, self.x_min, self.x_max)

                    self.state = new_point
                else:
                    # If the gradient is zero, generate a random direction vector
                    direction = np.random.normal(loc=0, scale=1, size=self.dimension)
                    direction /= np.linalg.norm(direction)  # Normalize the direction vector

                    # Calculate the step size based on the exploration range, step size factor, and no improvement duration
                    step_size = exploration_range * step_size_factor / (1 + 0.1 * self.no_improvement_duration)

                    # Generate a new point by taking a small step in the random direction
                    new_point = best_point + step_size * direction

                    # Ensure the new point stays within the valid range
                    new_point = np.clip(new_point, self.x_min, self.x_max)

                    self.state = new_point
            else:
                # If no best points are available, start with random initialization
                self.state = np.random.uniform(low=self.x_min, high=self.x_max, size=self.dimension)

            return self.state, {}

        else:
            self.state = np.random.uniform(low=self.x_min, high=self.x_max, size=self.dimension)
            return self.state, {}
            # if self.best_action_so_far is not None:
            #     if self.local_max - self.local_min > (self.global_max - self.global_min) * 0.5:
            #         self.local_min, self.local_max = self.global_min, self.global_max
            #         self.state = np.random.uniform(self.local_min, self.local_max, size=self.dimension)
            #     return self.state, {}
            # else:
            #     self.state = np.random.uniform(low=self.x_min, high=self.x_max, size=self.dimension)
            #     # print(f"Random new state: {self.state}")
            #     return self.state, {}


def learning_rate_func(progress_remaining: float) -> float:
    """
    This function takes the remaining progress (from 1 to 0) and returns the learning rate.
    You can modify this function to implement your own learning rate schedule.
    """
    start_lr = 0.01  # Starting learning rate
    end_lr = 0.000001  # Final learning rate
    lr = start_lr * progress_remaining + end_lr * (1 - progress_remaining)
    return lr


def ent_coef_func(progress_remaining: float) -> float:
    """
    This function takes the remaining progress (from 1 to 0) and returns the ent_coef.
    You can modify this function to implement your own ent_coef schedule.
    """
    start_ent_coef = 0.8  # Starting ent_coef
    end_ent_coef = 0.01  # Final ent_coef
    ent_coef = start_ent_coef * progress_remaining + end_ent_coef * (1 - progress_remaining)
    return ent_coef


def normal_train_eval(epochs: float, dimension: int, model: str, verbose: bool = False, scaling: str = "normal"):
    import time

    # Create environment
    env = CustomEnv(dimension=dimension, scaling=scaling)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.7 * np.ones(n_actions))

    # Create model
    if model == "DDPG":
        model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
                     buffer_size=1000)
    elif model == "SAC":
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
                    learning_starts=1000)
    elif model == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.5, learning_rate=0.0001,
                    vf_coef=0.5)

    if verbose:
        logger.info("Model created")

        # Train model
        logger.info(f"Start training. Epochs: {epochs}; Model: {model}; Dimension: {dimension}")
    start_training_time = time.time()
    model.learn(total_timesteps=int(epochs), progress_bar=False)
    if verbose:
        print("---" * 100)
        print(f"\nTraining time: {round(time.time() - start_training_time, 3)}s")
        print(f"Residuum Average: {np.mean(env.distances)}")
        print("===" * 100)

    if verbose:
        if dimension == 1:
            # plot residuum
            plot_nse(env.nse(), np.linspace(-5, 5, 10000), dimension=dimension, contour_plot=False,
                     points=env.good_points_plot,
                     points_x=None, distances=env.best_distances)

            # plot thresholds
            if len(env.good_points_thrs_list) > 0:
                x_thrs = np.arange(0, len(env.good_points_thrs_list))
                plt.plot(x_thrs, env.good_points_thrs_list, marker="o", markersize=2, label="Thresholds")
                plt.legend()
                plt.grid()
                plt.ylabel("Threshold")
                plt.xlabel("Epoch")
                plt.title("Threshold over Epoch")
                plt.show()

            # plot improvementrate history
            plt.plot(env.improvement_rate_history)
            plt.grid()
            plt.ylabel("Improvement Rate")
            plt.xlabel("Epoch")
            plt.show()

        if dimension == 2:
            print(f"Len of good points plot: {len(env.good_points_plot)}")
            plot_nse(env.nse(), np.linspace(-5, 5, 100), dimension=dimension, contour_plot=True,
                     points=env.good_points_plot,
                     points_x=None, distances=env.best_distances)

    if verbose:
        time_residuum_map = env.time_residuum
        time_action_map = env.time_action
        if dimension == 1:
            # fig, axs = plt.subplots(2, 1, figsize=(15, 9))
            # times_residuum = [time for time, residuum in time_residuum_map]
            # residuums = [residuum for time, residuum in time_residuum_map]
            # times_action = [time for time, action in time_action_map]
            # actions = [action for time, action in time_action_map]
            #
            # axs[0].plot(times_residuum, residuums, label=f"Residuum for NSE 1")
            # axs[0].set_title("Residuum over time for NSE 1")
            # axs[0].set_xlabel("Time (in s)")
            # axs[0].set_ylabel("Residuum")
            # axs[0].grid()
            #
            # # scale logarithmic
            # axs[0].set_yscale('log')
            #
            # axs[1].plot(times_action, actions)
            # axs[1].set_title("Time Action")
            # axs[1].set_xlabel("Time")
            # axs[1].set_ylabel("Action")

            plt.show()

            # print reward action mapping for all actions in matplotlib
            plt.figure(figsize=(8, 6))
            actions = [action for action, reward in env.reward_action_map]
            rewards = [reward for action, reward in env.reward_action_map]
            plt.plot(actions, rewards, "o", label=f"Reward Action Mapping, Scaling: {env.scaling}", markersize=1.5,
                     alpha=0.6)
            # plot global optima as red vertical lines
            global_optimums = env.global_optimum
            for idx, optimum in enumerate(global_optimums):
                if idx == 0:
                    plt.axvline(x=optimum, color='red', linestyle='--', label='Global Optimum')
                else:
                    plt.axvline(x=optimum, color='red', linestyle='--')

            plt.xlabel('Action')
            plt.ylabel('Reward')
            # plt.ylim(-1, 1)
            plt.xlim(-5, 5)
            plt.title('Reward Action Mapping')
            plt.grid()
            plt.legend()
            plt.show()

            # stats of rewards
            print(f"Mean reward: {np.mean(rewards)}")
            print(f"Max reward: {np.max(rewards)}")
            print(f"Min reward: {np.min(rewards)}")
            print(f"Std reward: {np.std(rewards)}")

            # plot time reward
            # plt.figure(figsize=(8, 6))
            # times_reward = [time for time, reward in env.time_reward]
            # rewards = [reward for time, reward in env.time_reward]
            # plt.scatter(times_reward, rewards, label="Reward for NSE 1", alpha=0.7, s=5)
            # plt.title("Reward over time for NSE 1")
            # plt.xlabel("Time (in s)")
            # plt.ylabel("Reward")
            # plt.grid()
            # plt.show()


        # elif dimension == 2:
        #     times_residuum = [time for time, residuum in time_residuum_map]
        #     residuums = [residuum for time, residuum in time_residuum_map]
        #
        #     plt.plot(times_residuum, residuums, label="Residuum for NSE 2")
        #     plt.title("Residuum over time for NSE 2")
        #     plt.xlabel("Time (in s)")
        #     plt.ylabel("Residuum")
        #     plt.grid()
        #     plt.yscale('log')
        #     plt.show()

        elif dimension > 2:
            times_residuum = [time for time, residuum in time_residuum_map]
            residuums = [residuum for time, residuum in time_residuum_map]

            plt.plot(times_residuum, residuums, label="Residuum for NSE 3")
            plt.title("Residuum over time for NSE 3")
            plt.xlabel("Time (in s)")
            plt.ylabel("Residuum")
            plt.grid()
            plt.yscale('log')
            plt.show()

    # good_points = env.good_points
    distances = env.distances
    #
    # print(f"Number of good points: {len(good_points)}")
    # print(f"Good points: {good_points}")

    print(f"Residuum Average: {np.mean(distances)}")

    x_min = -3.0
    x_max = 3.0
    y_min = -3.0
    y_max = 3.0

    # plot improvement rate history
    # plt.figure(figsize=(8, 6))
    # plt.plot(env.improvement_rate_history)
    # plt.xlabel('Step')
    # plt.ylabel('Improvement Rate')
    # plt.title('Improvement Rate History')
    # plt.grid()
    # plt.show()


def benchmark(epochs: list, dimension=1):
    df_list = []

    for epoch in epochs:
        print(f"Epoch: {epoch}")
        env = CustomEnv(dimension=dimension)

        # action noise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.75 * np.ones(n_actions))
        # model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
        #             buffer_size=1000)

        # model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
        #             learning_starts=1000)
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.5, vf_coef=0.5)
        # logger.info(f"Model created; Epochs: {epoch}")

        # logger.info("Start training")
        start_training_time = time.time()
        model.learn(total_timesteps=int(epoch), progress_bar=False, tb_log_name=f"benchmark_{epoch}")
        training_time = round(time.time() - start_training_time, 3)
        print("---" * 100)
        print(f"\nTraining time: {training_time}s")
        print("===" * 100)

        best_action = env.best_actions[-1]
        best_residuum = env.best_residuum
        action_distance_global_optimum = {}
        if dimension > 1:
            for action, residuum in env.action_precision.items():
                action_distance_global_optimum[action] = np.linalg.norm(np.array(action) - np.array(env.global_optimum))
            best_action_global_optimum = min(action_distance_global_optimum, key=action_distance_global_optimum.get)
            best_residuum_global_optimum = action_distance_global_optimum[best_action_global_optimum]
            best_action_global_optimum = list(best_action_global_optimum)
            best_action_global_optimum = np.array(best_action_global_optimum)
            print(f"Residuum Average: {np.mean(env.distances)}")

            df_epoch = pd.DataFrame({'Epochs': [epoch],
                                     'Time': [training_time],
                                     'Best Action': [best_action],
                                     'Best Residuum': [best_residuum],
                                     'Best Action global optimum': [best_action_global_optimum],
                                     'Best Residuum global optimum': [best_residuum_global_optimum]})
            df_list.append(df_epoch)
        else:
            print(f"Residuum Average: {np.mean(env.distances)}")
            df_epoch = pd.DataFrame({'Epochs': [epoch],
                                     'Time': [training_time],
                                     'Best Action': [best_action],
                                     'Best Residuum': [best_residuum]})
            df_list.append(df_epoch)

    df = pd.concat(df_list, ignore_index=True)
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"benchmark_{date}.csv", index=False)


if __name__ == '__main__':
    epochs = 8e3
    dimension = 1  # first and simplest nse
    # dimension = 2  # rosenbrock function
    # dimension = 10 # 10-dimensional nse (Economics Modeling Problem)
    model = "PPO"
    scaling = "normal"
    # start from 0.1 to 0.95 with 0.05 steps
    normal_train_eval(epochs, dimension, model, verbose=True, scaling=scaling)

    # for i in tqdm(range(5)):
    #     normal_train_eval(epochs, dimension, model)
    # print("---" * 100)
    # benchmark([1e3, 5e3, 1e4, 5e4], dimension=dimension)

    # plot reward action map
