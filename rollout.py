from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.metrics import mean_squared_error
import numpy as np

def rollout_with_policy_and_score(model, policy, domain_name, start_state, n_steps, true_final_state, refiners):
    """
    Rolls out a transition model for n steps using the given policy to generate actions.

    Parameters:
        model: FaultyTransitionModel with .predict(state, action) → np.ndarray
        policy: Trained stable-baselines3 policy with .predict(obs)
        domain_name: Gym domain name (e.g., "Acrobot-v1")
        start_state: np.ndarray
        n_steps: Number of steps to simulate
        true_final_state: Ground truth state to compare against
        refiners: dict of {domain_name: refiner function}

    Returns:
        (mse, predicted_final_state)
    """
    state = np.asarray(start_state)
    normalized_domain = domain_name.replace('-', '_')

    if normalized_domain not in refiners:
        raise KeyError(f"[rollout] No refiner registered for domain: {normalized_domain}")

    refiner = refiners[normalized_domain]

    for step in range(n_steps):
        try:
            refined = refiner(state)
        except Exception as e:
            raise ValueError(f"[rollout] Failed to refine state at step {step}. State: {state}. Error: {e}")

        action, _ = policy.predict(refined, deterministic=True)

        try:
            next_state = model.predict(state, int(action))
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            state = np.asarray(next_state)
        except Exception as e:
            raise ValueError(f"[rollout] Model prediction failed at step {step} with action {action}. Error: {e}")

    predicted_final_state = state

    try:
        mse = mean_squared_error(true_final_state, predicted_final_state)
    except Exception as e:
        raise ValueError(f"[rollout] MSE computation failed. True: {true_final_state}, Predicted: {predicted_final_state}. Error: {e}")

    return mse, predicted_final_state



def infer_fault_from_segment(models_by_fault, policy, domain_name, start_state, n_steps, true_final_state, refiners):
    """
    Runs rollout_with_policy_and_score on each model to infer the most likely fault mode.

    Args:
        models_by_fault: dict {fault_mode_str: FaultyTransitionModel}
        policy: Trained Gym-compatible policy
        domain_name: Gym domain name
        start_state: np.ndarray
        n_steps: int
        true_final_state: np.ndarray
        refiners: dict of refiners per domain

    Returns:
        best_fault_mode (str), best_mse (float), all_mses (dict)
    """
    best_mse = float('inf')
    best_fault_mode = None
    all_mses = {}

    for fault_mode, model in models_by_fault.items():
        mse, _ = rollout_with_policy_and_score(
            model, policy, domain_name,
            start_state, n_steps,
            true_final_state,
            refiners
        )
        all_mses[fault_mode] = mse
        if mse < best_mse:
            best_mse = mse
            best_fault_mode = fault_mode

    return best_fault_mode, best_mse, all_mses



import os
from stable_baselines3 import PPO, DQN, A2C  # extend this if needed
from rl_models import models  # your global registry: {'PPO': PPO, ...}

def load_policy(domain_name, ml_model_name, render_mode=None):
    """
    Loads a trained RL policy from disk for a given Gym domain.

    Args:
        domain_name (str): e.g., "CartPole-v1"
        ml_model_name (str): e.g., "PPO", "DQN", etc.
        render_mode (str or None): Rendering mode (optional)

    Returns:
        model: loaded policy (e.g., PPO object)
        env: Gym environment with wrapper
    """
    import gym
    from wrappers import wrappers

    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))

    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = os.path.join(models_dir, f"{domain_name}__{ml_model_name}.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model_class = models[ml_model_name]
    model = model_class.load(model_path, env=env)

    return model, env
