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

    def _prepare_data(self, data):
        X, Y = [], []
        for s, a, s_prime in data:
            x = np.append(s, a)
            y = s_prime  # or: y = s_prime - s for delta prediction
            X.append(x)
            Y.append(y)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def _train_model(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)

        if self.model_type == 'mlp':
            self.model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
        else:
            self.model = LinearRegression()

        self.model.fit(X_train, Y_train)
        Y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(Y_test, Y_pred)

    def predict(self, state, action):
        x = np.append(state, action).reshape(1, -1)
        return self.model.predict(x)

    def score(self):
        return self.mse

    def __repr__(self):
        return f"FaultyTransitionModel(fault_mode={self.fault_mode}, model_type={self.model_type}, MSE={self.mse:.4f})"
