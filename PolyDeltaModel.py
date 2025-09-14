import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class PolyDeltaModel:
    """Tiny nonlinear regressor: fits Δs = s_next - s, predicts s_next = s + Δs"""
    def __init__(self, fault_mode, degree=3, alpha=3.0):
        self.fault_mode = fault_mode
        self.degree = degree
        self.alpha = alpha
        self.pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha, fit_intercept=True, random_state=0),
        )
        self.model_type = "nonlinear"
        self.obs_dim = None
        self.mse_ = None
        # metadata cached at fit-time
        self.n_samples_ = 0
        self.input_dim_ = 0
        self.output_dim_ = 0
        self._fitted = False

    def fit(self, data):
        # data: list of (s, a, s_next) for a SINGLE action
        S  = np.array([np.atleast_1d(s).astype(np.float32)  for s, a, sp in data])
        SP = np.array([np.atleast_1d(sp).astype(np.float32) for s, a, sp in data])
        self.obs_dim = S.shape[1]
        Y = SP - S
        self.pipe.fit(S, Y)
        self.mse_ = float(mean_squared_error(Y, self.pipe.predict(S)))
        # cache metadata
        self.n_samples_ = int(S.shape[0])
        self.input_dim_ = int(S.shape[1])
        self.output_dim_ = int(Y.shape[1] if Y.ndim > 1 else 1)
        self._fitted = True
        return self

    def predict(self, state):
        s = np.asarray(state, np.float32).ravel()
        if self.obs_dim is not None and s.size != self.obs_dim:
            raise ValueError(f"state dim {s.size} != expected {self.obs_dim}")
        delta = self.pipe.predict(s[None, :]).ravel()
        return s + delta  # next state

    def get_metadata_string(self):
        if not self._fitted:
            return (f"🧠 Model Type: {self.model_type} | Fault Mode: {self.fault_mode} | "
                    f"Status: not fitted")
        return (f"🧠 Model Type: {self.model_type} | Fault Mode: {self.fault_mode} | "
                f"deg={self.degree}, alpha={self.alpha} | "
                f"Training Samples: {self.n_samples_} | "
                f"Input Dim: {self.input_dim_}, Output Dim: {self.output_dim_} | "
                f"MSE: {self.mse_:.4f}")
