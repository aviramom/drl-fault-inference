from sklearn.metrics import mean_squared_error
import numpy as np
import numpy as np

import no_faults_executor as nfe
import with_faults_executor as wfe
from FaultyTransitionModel import FaultyTransitionModel
from evaluation import evaluate_model_on_faults
from evaluation import evaluate_model_on_testset
from fault_mode_generators import FaultModeGeneratorDiscrete
from pipeline import read_json_data
from pipeline import separate_trajectory
from Faulty_Data_Extractor import get_faulty_data, get_augmented_faulty_data, get_all_transitions_under_fault
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


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



def train_models_for_fault_modes(
    domain_name,
    model_name,
    all_fault_modes,
    fault_mode_generator,
    num_trajectories,
    debug_mode,
    fault_probability,
    render_mode,
    max_exec_len,
    get_transitions_func,
    model_type='linear'
):
    """
    Trains a model per fault mode and returns a dictionary of trained models.

    Args:
        domain_name (str): Name of the Gym domain (e.g., 'CartPole_v1').
        model_name (str): Name of the SB3 model to use.
        all_fault_modes (list): List of fault mode names.
        fault_mode_generator (FaultModeGenerator): Fault mode generator instance.
        num_trajectories (int): Number of trajectories to collect for training.
        debug_mode (bool): Whether to enable debugging prints.
        fault_probability (float): Fault injection probability (e.g., 1.0 for always fault).
        render_mode (str): Render mode for the env (e.g., 'rgb_array').
        max_exec_len (int): Maximum steps per trajectory.
        get_transitions_func (callable): Function to get transitions under fault mode.
        model_type (str): Model type to use for training ('linear', 'mlp').

    Returns:
        dict: A dictionary mapping each fault mode to its trained FaultyTransitionModel.
    """
    models_by_fault = {}

    for fault_mode in all_fault_modes:
        print(f"\n==== Training model for fault mode: {fault_mode} ====")

        # Get transitions under this fault mode
        trajectory_data = get_transitions_func(
            num_trajectories,
            domain_name,
            debug_mode,
            fault_mode,
            fault_probability,
            render_mode,
            model_name,
            fault_mode_generator,
            max_exec_len
        )

        print(f"Total training samples (all transitions under fault): {len(trajectory_data)}")

        # Train/test split
        train_data, test_data = train_test_split(trajectory_data, test_size=0.2, random_state=42)
        print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")

        # Train model
        model = FaultyTransitionModel(fault_mode=fault_mode, data=train_data, model_type=model_type)
        models_by_fault[fault_mode] = model

        # Evaluate model
        evaluate_model_on_testset(model, test_data)
        # evaluate_model_on_faults(
        #     model=model,
        #     domain_name=domain_name,
        #     fault_mode_name=fault_mode,
        #     fault_mode_generator=fault_mode_generator,
        #     ml_model_name=model_name,
        #     num_samples=2000,
        #     render_mode=render_mode
        # )

    print("\n✅ All models trained and evaluated.")
    return models_by_fault





