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
    eq2 = lambda x: np.sin(2 * x)

    x = np.linspace(x_min, x_max, num_points)
    # x = x[x != 0]

    yeq1 = np.vectorize(eq1)
    yeq2 = np.vectorize(eq2)

    y_arrayeq1 = yeq1(x)
    y_arrayeq2 = yeq2(x)
    # point = x
    res = (eq1(x) - eq2(x)) ** 2

    return res


def nse():
    """
    System of nonlinear equations.
    :return: Numpy array of equations.
    """
    eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
    eq2 = lambda x: np.sin(2 * x)

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


def plot_nse(equations, x_array, points=None, points_x=None, distances=None):
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

    axs[0].set_ylim(-3.0, 3.0)
    axs[0].set_xlim(-3.0, 3.0)
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
        self.y_min = -5.0
        self.y_max = 5.0
        # self.action_space = gym.spaces.Box(low=np.array([self.x_min, self.y_min]),
        #                                   high=np.array([self.x_max, self.y_max]),
        #                                   dtype=np.float32)
        self.action_space = gym.spaces.Box(low=self.x_min, high=self.x_max)  # action is just x value

        self.observation_space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(2,))
        self.state = np.array(
            [random.uniform(self.x_min, self.x_max),
             random.uniform(self.y_min, self.y_max)])  # self.state = point in space
        self.nse = nse
        self.best_actions = []
        self.last_obs = None
        self.actions = []
        self.distances = []
        self.best_distances = []
        self.good_points = []
        self.good_points_plot = []
        self.good_res_plot = []
        self.best_action_so_far = None
        self.good_points_thrs = 0.9
        self.rewards = []
        self.reward_action_map = []

        self.consecutive_no_improvement = 0

        self.min_residuum = float('inf')
        self.max_residuum = float('-inf')

    def calculate_derivative(self, action):
        d_eq1 = lambda x: 5 * x ** 4 - 12 * x ** 3 + 3 * x ** 2 + x
        d_eq2 = lambda x: 2 * np.cos(2 * x)

        return np.array([d_eq1(action), d_eq2(action)])

    def get_distance(self, point):
        """
        Calculate the distance between a point and each equation in the system of nonlinear equations.
        :param point: Tuple containing the coordinates of the point.
        :return: Numpy array of distances between the point and each equation.
        """
        return np.array([distance_func_continuous(point, eq) for eq in self.nse])

    def get_distance_discrete(self, point):
        eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
        eq2 = lambda x: np.sin(2 * x)

        # Berechnung der Funktionswerte
        value_eq1 = eq1(point)
        value_eq2 = eq2(point)

        # Logarithmische Skalierung der Funktionswerte
        log_base = 10  # Basis des Logarithmus (anpassbar)
        scaled_eq1 = np.log(abs(value_eq1) + 1) / np.log(log_base)
        scaled_eq2 = np.log(abs(value_eq2) + 1) / np.log(log_base)

        # Vorzeichenkorrektur für negative Werte
        if value_eq1 < 0:
            scaled_eq1 = -scaled_eq1
        if value_eq2 < 0:
            scaled_eq2 = -scaled_eq2

        res = (scaled_eq1 - scaled_eq2) ** 2

        res_old = (eq1(point) - eq2(point)) ** 2

        # Min-Max-Skalierung der Ergebnisse von eq1 und eq2
        # min_val = -50.0
        # max_val = 50.0
        # scaled_eq1 = (eq1(point) - min_val) / (max_val - min_val)
        # scaled_eq2 = (eq2(point) - min_val) / (max_val - min_val)
        #
        # res = (scaled_eq1 - scaled_eq2) ** 2

        return np.tanh(res_old)

    def _plot_res(self):
        eq1 = lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2
        eq2 = lambda x: np.sin(2 * x)

        x = np.linspace(-1, 2.74, 1000)

        # Berechnung der Funktionswerte
        value_eq1 = eq1(x)
        value_eq2 = eq2(x)

        log_bases = [5, 10]

        for log_base in log_bases:
            # Logarithmische Skalierung der Funktionswerte
            scaled_eq1 = np.log(abs(value_eq1) + 1) / np.log(log_base)
            scaled_eq2 = np.log(abs(value_eq2) + 1) / np.log(log_base)

            # Vorzeichenkorrektur für negative Werte
            scaled_eq1[value_eq1 < 0] = -scaled_eq1[value_eq1 < 0]
            scaled_eq2[value_eq2 < 0] = -scaled_eq2[value_eq2 < 0]

            res = (scaled_eq1 - scaled_eq2) ** 2

            res_old = (eq1(x) - eq2(x)) ** 2

            # plot res and res old
            plt.plot(x, res, label=f"res log base {log_base}")

        plt.grid()
        #plt.plot(x, res_old, label="res normal")
        plt.plot(x, np.tanh(res_old), label="tanh(res normal)")
        plt.plot(x, np.clip(res_old, 0, 1), label="clip(res normal)")
        plt.legend()

        plt.show()

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

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action to take. (Tuple containing the coordinates of the point)
        :return: state, reward, done, info
        """

        discrete = True
        self.actions.append(action)
        if discrete:
            self.distances.append(self.get_distance_discrete(action))

        # just print better action
        if self.distances[-1] <= min(self.distances):
            print("Best action:", self.actions[-1])
            print("Residuum:", self.distances[-1])
            print()
            self.best_actions.append(action)
            self.best_distances.append(self.distances[-1])

        if self.distances[-1] <= self.good_points_thrs and len(self.good_points) > 0:
            print(f"Good point: {self.good_points[-1]}\tResiduum: {self.distances[-1]}")
            self.good_res_plot.append(self.distances[-1])

        # add points that are better than a given threshold just for plotting
        if self.distances[-1] <= 1e-5:
            self.good_points_plot.append(action)

        if discrete:
            self.state = action

            residuum = self.get_distance_discrete(action)

            self.min_residuum = min(self.min_residuum, residuum)
            self.max_residuum = max(self.max_residuum, residuum)

            if self.max_residuum - self.min_residuum != 0:
                normalized_residuum = (residuum - self.min_residuum) / (self.max_residuum - self.min_residuum)
            else:
                normalized_residuum = 0

            # reward = np.exp(-normalized_residuum * 100)
            reward = 1 - residuum

            if isinstance(reward, int) or isinstance(reward, float):
                self.rewards.append(reward)
                self.reward_action_map.append((action, reward))
            else:
                self.rewards.append(reward[0])
                self.reward_action_map.append((action, reward[0]))

            if residuum > self.good_points_thrs:
                reward -= 0.5

            # print(f"Reward: {reward}")

            if residuum <= self.good_points_thrs:
                self.good_points.append(action)

            # Reward if action is better than last action
            if len(self.distances) > 1 and residuum < self.distances[-2]:
                reward += 0.10
                self.best_action_so_far = action

            # penalty if action is worse than last action
            if len(self.best_actions) > 1 and residuum > self.best_distances[-2]:
                reward -= 0.5

            done = False
            truncated = False
            reward = np.clip(reward, 0, 1)

            epsilon = 1e-8
            max_residuum = 100.0  # Determine the maximum possible residuum value based on your problem

            max_reward = 1 / epsilon
            min_reward = 1 / (max_residuum + epsilon)

            if residuum <= self.good_points_thrs:  # if distance is less than reset
                # done = True
                # reward = 1 + (self.good_points_thrs - residuum) / self.good_points_thrs
                #
                # if len(self.best_distances) > 1:
                #     improvement_rate = (self.best_distances[-2] - self.best_distances[-1]) / self.best_distances[-2]
                #     if improvement_rate > 0.1:
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

                done = True
                #reward = -np.log(residuum + epsilon)
                #reward = (reward - min_reward) / (max_reward - min_reward)
                self.good_points_thrs *= max(0.9, self.good_points_thrs * 0.99)
            else:
                reward = -np.log(residuum + epsilon)
                reward = (reward - min_reward) / (max_reward - min_reward)

            # else:
            #     reward = 1 / (residuum + epsilon)
            #     reward = (reward - min_reward) / (max_reward - min_reward)

            # self.best_action.append(action)
            self.last_obs = self.state

            reward = np.clip(reward, 0, 1)



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
    env = CustomEnv(nse(), plot=False, discrete=True)
    logger.info("Environment created")

    env._plot_res()

    # check environment
    # check_env(env)

    # create model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, ent_coef=0.15)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.8 * np.ones(n_actions))
    # model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise,
    #             buffer_size=1000)

    model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, train_freq=1, action_noise=action_noise, learning_starts=1000)
    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, ent_coef=0.15)
    logger.info("Model created")

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
        model.learn(total_timesteps=int(3e3), progress_bar=True)

    _, distances = env.best_actions, env.distances
    good_points, good_points_plot = env.good_points, env.good_points_plot
    good_res_plot = env.good_res_plot
    rewards = env.rewards

    print(good_points)
    print(f"Number of good points: {len(good_points)}")
    # print(f"Good points: {good_points}")

    print(f"Residuum Average: {np.mean(distances)}")

    x_min = -3.0
    x_max = 3.0
    y_min = -3.0
    y_max = 3.0
    nse_system = nse()
    eqs = [lambda x: x ** 5 - 3 * x ** 4 + x ** 3 + 0.5 * x ** 2, lambda x: np.sin(2 * x)]
    good_points_y_values = [eqs[0](point) for point in good_points]
    good_points_xy = [(point, eqs[0](point)) for point in good_points]

    # plot best residuen
    best_distances = env.best_distances

    plot_nse(eqs, np.linspace(x_min, x_max, 100000), points=None, points_x=good_points_plot, distances=good_res_plot)

    # plot reward action mapping
    actions = [action for action, reward in env.reward_action_map]
    rewards = [reward for action, reward in env.reward_action_map]
    plt.scatter(actions, rewards, s=1.2, label="reward = exp(-residuum * 1000)")
    real_solutions = [-0.69983978673645688906, 0.0, 2.4936955491125650267]
    for real_sol in real_solutions:
        plt.axvline(x=real_sol, color='red', label='Real Solution')
    plt.xlabel("Action")
    plt.ylabel("Reward")
    plt.xlim(env.x_min, env.x_max)
    plt.legend()
    plt.show()

    # save model
    model.save("ddpg_nse")

    # tensorboard befehl: tensorboard --logdir ./tmp/
