import numpy as np
import json


class LinearRegression:
    """
    Linear Regression Model with Gradient Descent

    This class implements a linear regression model using gradient descent
    optimization for training.
    It provides methods for model initialization, training, prediction,
    and model persistence.

    Parameters:
        learning_rate (float): The learning rate used in gradient descent.
        convergence_tol (float, optional): The tolerance for convergence
        (stopping criterion). Defaults to 1e-6.

    Attributes:
        W (numpy.ndarray): Coefficients (weights) for the linear regression
        model.
        b (float): Intercept (bias) for the linear regression model.

    Examples:
        >>> from linear_regression import LinearRegression
        >>> model = LinearRegression(learning_rate=0.01)
        >>> model.fit(X_train, y_train, iterations=1000)
        >>> predictions = model.predict(X_test)
    """
    def __init__(self, learning_rate, convergence_tol=1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None

    def initialize_parameters(self):
        """
        Initialize model parameters.
        """

        self.W = np.random.randn() * 0.01
        self.b = 0

    def forward(self, X):
        """
        Compute the forward pass of the linear regression model.
        X (numpy.array): Input data of shape(m,).
        returns: numpy.ndarray: Predictions of shape(m,).
        """
        return np.dot(X, self.W) + self.b

    def compute_cost(self, predictions):
        """
        Compute the mean squared error cost.

        Parameters:
            predictions (numpy.ndarray): Predictions of shape (m,).

        Returns:
            float: Mean squared error cost.
        """
        m = len(predictions)
        cost = np.sum(np.square(predictions - self.y)) / (2 * m)
        return cost

    def backward(self, predictions):
        """
        Compute gradients for model parameters.
        prediction (numpy.ndarray): predictions of shape(m,).
        Updates:	numpy.ndarray: Gradient of W.
                    float: Gradient of b.
        """
        m = len(predictions)
        self.dW = np.dot(predictions - self.y, self.X) / m
        self.db = np.sum(predictions - self.y) / m

    def standardize_data(self, X_train):
        """
        Standardizes the input data using mean and standard deviation.

        Parameters:
            X_train (numpy.ndarray): Training data.

        Returns:
            numpy.ndarray of standardized training data.
        """
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        self.X = (X_train - mean) / std

    def fit(self, X, y, iterations):
        """
        Fit the linear regression mode t the training data.

        Parameters:
        X (numpy.ndarray): Training input data of shape(m,).
        y (numpy.ndarray): Training labels of shape (m,).
        iterations (int): The number of iterations for gradient descent.
        plot_cost (bool, optional): Whether to plot the cost during training.
        Default to True.
        Raises:
        AssertionError: If input data and labels are not NumPy arrays or have
        mismatched shapes.

        plots: Ploty line chart showing cost vs. iteration (if plot is True).
        """
        assert isinstance(X, np.ndarray), "X must be a NumPy array"
        assert isinstance(y, np.ndarray), "y must be a NumPy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number"
        "of samples"
        assert X.shape[0] > 1, "There should be more samples"
        assert iterations > 0, "Iterations must be greater than 0"

        self.y = y
        self.mean = np.mean(X)
        self.std = np.std(X)
        self.standardize_data(X)
        self.initialize_parameters()
        costs = []

        for i in range(iterations):
            predictions = self.forward(self.X)
            cost = self.compute_cost(predictions)
            self.backward(predictions)
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db
            costs.append(cost)

            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f'Converged after {i} iterations.')
                break

    def predict(self, X):
        """
        Predict target values for new input data.
        """
        return self.forward(X)

    def denormalize_thetas(self):

        self.W = self.W / self.std
        self.b = self.b - self.W * self.mean

    def save_thetas(self, filename=None):
        """
        Save the trained model to thetas.json.
        """
        self.denormalize_thetas()

        model_data = {
            'theta1': self.W,
            'theta0': self.b
            }
        with open(filename, 'w') as file:
            json.dump(model_data, file)
