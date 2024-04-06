import numpy as np
import pandas as pd
import numdifftools as nd
import time


def newton_raphson_scalar(func, x0, tol=1e-9, max_iter=100):
    x = x0
    for _ in range(max_iter):
        # Compute the function value and its gradient at the current point
        f = func(x)
        gradient = nd.Gradient(func)(x)

        # Check if the gradient is zero
        if np.allclose(gradient, 0):
            print("Warning: Newton-Raphson method did not converge. Gradient is close to zero.")
            break

        # Compute the next approximation using Newton-Raphson iteration
        x1 = x - f / gradient

        # calculate tolerance
        tol = np.linalg.norm(x1 - x)

        # print(f"Iteration: {_}, Solution: {x1}")

        # Update x
        x = x1

        # Check if the absolute value of the function value is less than the tolerance
        # if np.abs(f) < tol:
        #     return x, _

        # Check if maximum number of iterations is reached
        if _ == max_iter - 1:
            return x, _, tol

    return x, _, tol


def newton_raphson(func, x0, tol=1e-8, max_iter=100):
    """
    Solve a system of nonlinear equations using the Newton-Raphson method.

    Args:
        func (callable): A function that takes a NumPy array of size n and returns a NumPy array of size n.
        x0 (numpy.ndarray): The initial guess for the solution, a NumPy array of size n.
        tol (float, optional): The tolerance for convergence. Default is 1e-8.
        max_iter (int, optional): The maximum number of iterations. Default is 100.

    Returns:
        numpy.ndarray: The solution to the system of nonlinear equations.
        int: The number of iterations required to converge.
    """
    x = x0.copy()
    n = len(x)

    for i in range(max_iter):
        # Compute the function values at the current point
        f_x = func(x)

        # Compute the Jacobian matrix using numdifftools
        jac = nd.Jacobian(func)(x)

        # Solve the linear system J(x) * dx = -f(x) for dx
        dx = np.linalg.solve(jac, -f_x)

        # Update the solution
        x += dx

        # print(f"Iteration: {i}, Solution: {x}")

        # Calculate the tolerance
        tol = np.linalg.norm(dx)

        # # Check for convergence
        # if np.linalg.norm(f_x) < tol:
        #     return x, i + 1

    return x, max_iter, tol


def f(x):
    return x ** 5 - 3 * x ** 4 + x ** 3 + (0.5) * x ** 2 - np.sin(2 * x)


def rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def economics_modeling_problem():
    n = 10

    def combined_eq(x):
        result = np.zeros(n)
        for k in range(1, n):
            summe = sum(x[i] * x[i + k] for i in range(0, n - k - 1))
            eq_k = (x[k - 1] + summe) * x[n - 1]
            result[k - 1] = eq_k

        eq2 = sum(x[l] for l in range(0, n)) + 1
        result[n - 1] = eq2

        return result

    return combined_eq


def economics_modeling_problem_1():
    list_of_eqs = list()
    n = 10
    for k in range(1, n):
        def eq(x, k=k):
            summe = sum(x[i] * x[i + k] for i in range(0, n - k - 1))
            return (x[k - 1] + summe) * x[n - 1]

        list_of_eqs.append(eq)

    eq2 = lambda x: sum(x[l] for l in range(0, n)) + 1
    list_of_eqs.append(eq2)

    return list_of_eqs


def newton_raphson_economics_modeling(funcs, x0, tol=1e-8, max_iter=100):
    """
        Solve a system of nonlinear equations using the Newton-Raphson method.

        Args:
            funcs (list of callables): A list of functions, each representing an equation of the system.
            x0 (numpy.ndarray): The initial guess for the solution, a NumPy array of size n.
            tol (float, optional): The tolerance for convergence. Default is 1e-8.
            max_iter (int, optional): The maximum number of iterations. Default is 100.

        Returns:
            numpy.ndarray: The solution to the system of nonlinear equations.
            int: The number of iterations required to converge.
        """
    x = x0.copy()
    n = len(x)
    eq_system = economics_modeling_problem()

    for i in range(max_iter):
        # Compute the function values at the current point
        f_x = np.array([func(x) for func in funcs])

        # Compute the Jacobian matrix using numdifftools
        jac = nd.Jacobian(eq_system)(x)
        jac_inv = np.linalg.inv(jac)

        # Solve the linear system J(x) * dx = -f(x) for dx
        dx = np.dot(jac_inv, -f_x)

        # Update the solution
        x += dx

        # Calculate the tolerance
        tol = np.linalg.norm(dx)

        # Check for convergence
        # if tol < tol:
        #     return x, i + 1

    return x, max_iter, tol


def benchmark_newton(x0, steps: list):
    results_df = pd.DataFrame(columns=['Steps', 'x0', 'Solution', 'Num Iter', 'Tolerance', 'Time'])

    for step in steps:
        start_time = time.time()
        # scalar
        # solution, num_iter, tol = newton_raphson_scalar(f, x0, max_iter=step)

        # vector
        solution, num_iter, tol = newton_raphson(rosenbrock, x0, max_iter=step)
        end_time = time.time()
        df = pd.DataFrame({'Steps': [step],
                           "x0": [x0],
                           'Solution': [solution],
                           'Num Iter': [num_iter],
                           'Tolerance': [tol],
                           'Time': [end_time - start_time]})
        # Append the current iteration DataFrame to the results DataFrame
        results_df = pd.concat([results_df, df], ignore_index=True)

        # save df as csv
    results_df.to_csv("benchmark_newton_results_nse2.csv", index=False)


if __name__ == "__main__":
    # x0 = 5.0  # Initial guess
    # steps = [5, 10, 15, 20]
    # benchmark_newton(x0, steps)

    # x0 = np.array([-5.0, 5.0])
    # steps = [5, 10, 15, 20]
    # benchmark_newton(x0, steps)

    # Define the list of equations for the economics modeling problem
    list_of_eqs = economics_modeling_problem_1()

    # Initial guess
    # random numbers between -100 and 100
    x0 = np.random.uniform(-100, 100, 10)

    # Solve the system of equations using Newton-Raphson method
    solution, num_iter, tol = newton_raphson_economics_modeling(list_of_eqs, x0, max_iter=3)

    print("Solution:", solution)
    print("Number of iterations:", num_iter)
    print("Tolerance:", tol)
