from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

class FaultyTransitionModel:
    def __init__(self, fault_mode, data, model_type='linear'):
        """
        fault_mode: label or descriptor for the fault (e.g., [1,1,2])
        data: list of (state, action, next_state) tuples
        model_type: 'linear', 'mlp', etc.
        """
        self.fault_mode = fault_mode
        self.model_type = model_type
        self.model = None

        self._prepare_data(data)
        self._train_model()

    def _prepare_data1(self, data):
        X, Y = [], []
        for s, a, s_prime in data:
            x = s
            y = s_prime  # or: y = s_prime - s for delta prediction
            X.append(x)
            Y.append(y)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def _prepare_data(self, data):
        X, Y = [], []
        for s, a, s_prime in data:
            # Convert scalar state to 1D array if needed
            s_array = np.atleast_1d(s)
            s_prime_array = np.atleast_1d(s_prime)
            x = s_array
            y = s_prime_array
            X.append(x)
            Y.append(y)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def _train_model(self):
        if self.model_type == 'mlp':
            self.model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
        else:
            self.model = LinearRegression()

        self.model.fit(self.X, self.Y)
        Y_pred = self.model.predict(self.X)
        self.mse = mean_squared_error(self.Y, Y_pred)

    def predict(self, state):
        x = np.atleast_2d(state)
        return self.model.predict(x)

    def score(self):
        return self.mse

    def plot_regression_for_dim(self, dim=0):
        """
        Plots predicted vs actual values for a specific output dimension,
        and prints the regression equation if model is linear.
        """
        if self.model_type != 'linear':
            print("⚠️ Plotting is only supported for linear models.")
            return

        Y_pred = self.model.predict(self.X)
        actual = self.Y[:, dim]
        predicted = Y_pred[:, dim]

        # Plot
        plt.figure(figsize=(6, 4))
        plt.scatter(actual, predicted, alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Ideal')
        plt.xlabel(f'Actual Dimension {dim}')
        plt.ylabel(f'Predicted Dimension {dim}')
        plt.title(f'Linear Regression (Dim {dim})\nFault Mode: {self.fault_mode}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print regression equation
        coef = self.model.coef_[dim]
        intercept = self.model.intercept_[dim]
        terms = [f"{coef[i]:+.3f}*x{i}" for i in range(len(coef))]
        equation = " + ".join(terms) + f" {intercept:+.3f}"
        print(f"🧮 Regression Equation for output dim {dim}:\n  y = {equation}")

    def plot_regression_line_for_feature(self, feature_idx=0, output_dim=0):
        """
        Plots the regression line for a selected feature and output dimension (only for linear models).
        X-axis: input feature at index `feature_idx`
        Y-axis: predicted output at dimension `output_dim`
        """
        if self.model_type != 'linear':
            print("⚠️ Only supported for linear models.")
            return

        x_vals = self.X[:, feature_idx]
        y_vals = self.Y[:, output_dim]

        coef = self.model.coef_[output_dim][feature_idx]
        intercept = self.model.intercept_[output_dim]

        # Compute regression line
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = coef * x_line + intercept

        # Plot
        plt.figure(figsize=(6, 4))
        plt.scatter(x_vals, y_vals, alpha=0.6, label='Data points')
        plt.plot(x_line, y_line, 'r-', label=f'Linear fit: y = {coef:.3f} * x + {intercept:.3f}')
        plt.xlabel(f'Feature x[{feature_idx}]')
        plt.ylabel(f'Output dim {output_dim}')
        plt.title(f'Fault {self.fault_mode} | Regression Line (Output {output_dim} vs Feature {feature_idx})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all_feature_regressions(self, output_dim=0, max_plots=None):
        """
        Plots regression lines of all input features (X[:, i]) vs. output Y[:, output_dim].
        Only works for linear models.

        Args:
            output_dim (int): which output dimension to visualize (e.g., 0 for s'[0])
            max_plots (int or None): optionally limit number of features plotted
        """
        if self.model_type != 'linear':
            print("⚠️ Only supported for linear models.")
            return

        num_features = self.X.shape[1]
        if max_plots is not None:
            num_features = min(num_features, max_plots)

        cols = 3
        rows = (num_features + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 4 * rows))

        for i in range(num_features):
            x_vals = self.X[:, i]
            y_vals = self.Y[:, output_dim]
            coef = self.model.coef_[output_dim][i]
            intercept = self.model.intercept_[output_dim]
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = coef * x_line + intercept

            plt.subplot(rows, cols, i + 1)
            plt.scatter(x_vals, y_vals, alpha=0.6, label='Data')
            plt.plot(x_line, y_line, 'r-', label=f'y = {coef:.2f}x + {intercept:.2f}')
            plt.xlabel(f'x[{i}]')
            plt.ylabel(f'y[{output_dim}]')
            plt.title(f'Feature {i} vs Output {output_dim}')
            plt.title(f'Fault {self.fault_mode} | Regression Line (Output {output_dim} vs Feature x[{i}])')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def print_regression_equation(self, output_dim):
        """
        Prints the full regression equation for a selected output dimension.
        """
        if self.model_type != 'linear':
            print("⚠️ Equation printing only supported for linear models.")
            return

        coef = self.model.coef_[output_dim]
        intercept = self.model.intercept_[output_dim]
        terms = [f"{coef[i]:+.3f}*x{i}" for i in range(len(coef))]
        equation = " + ".join(terms) + f" {intercept:+.3f}"
        print(f"📘 Fault Mode: {self.fault_mode}, Output Dim {output_dim}")
        print(f"y = {equation}")



    def __repr__(self):
        return f"FaultyTransitionModel(fault_mode={self.fault_mode}, model_type={self.model_type}, MSE={self.mse:.4f})"
