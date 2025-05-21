import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from sklearn.metrics import mean_squared_error
import Faulty_Data_Extractor
from wrappers import wrappers
from collections import defaultdict
from fault_mode_generators import FaultModeGeneratorDiscrete
import with_faults_executor as wfe
from pipeline import separate_trajectory
from state_refiners import refiners
from rollout import infer_fault_from_segment, load_policy

def evaluate_model_on_faults(model, domain_name, fault_mode_name, fault_mode_generator,
                             ml_model_name, num_samples=100, render_mode=None):
    """
    Evaluates how well the learned model predicts faulty transitions:
    (state, action) -> faulty next_state, using randomly sampled (state, action) pairs
    with faults applied through the fault_mode_generator.

    Parameters:
        model (FaultyTransitionModel): The trained model to evaluate.
        domain_name (str): The Gym environment name.
        fault_mode_name (str): The fault mode string to use with the generator (e.g., "[1,1,2]").
        fault_mode_generator (FaultModeGenerator): Generator that returns a faulty action function.
        ml_model_name (str): Name of the model (used if needed for compatibility).
        num_samples (int): Number of (state, action) samples to evaluate.
        render_mode (str or None): Rendering mode for Gym.

    Returns:
        float: Mean squared error between true and predicted faulty next states.
    """
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    faulty_action_fn = fault_mode_generator.generate_fault_mode(fault_mode_name)
    mapping = eval(fault_mode_name)
    faulty_actions = [i for i, mapped in enumerate(mapping) if i != mapped]
    actual_states = []
    predicted_states = []

    for _ in range(num_samples):
        # Reset with a random seed
        seed = np.random.randint(0, 1_000_000)
        state, _ = env.reset(seed=seed)

        # Sample an action (intended)
        intended_action=random.choice(faulty_actions)

        # Apply fault mode to get actual faulty action
        faulty_action = faulty_action_fn(intended_action)

        # Try restoring env state to ensure deterministic step
        try:
            env.unwrapped.state = state
        except AttributeError:
            print("Environment does not support direct state setting.")
            continue

        # Step with faulty action and get true post-state
        next_state, _, done, trunc, _ = env.step(faulty_action)
        actual_states.append(next_state);
        if done or trunc:
            continue  # skip terminal transitions

        # Predict post-state using model
        predicted = model.predict(state, intended_action)[0]  # model sees only the planned action
        #print(f'actual state: {state}, predicted: {predicted}')
        predicted_states.append(predicted)
        # Compute MSE
        # mse = mean_squared_error(next_state, predicted)
        # errors.append(mse)


    env.close()
    actuals = np.array(actual_states)
    predictions = np.array(predicted_states)

    mse = np.mean((actuals - predictions) ** 2)
    var = np.var(actuals)

    print(f"[Synthetic, Unseen States]Overall MSE: {mse:.6f}, Variance of actuals: {var:.6f}, MSE / Var: {mse / var:.2%}\n")

    # ✅ Per-dimension evaluation
    if actuals.shape[1] > 1:
        print("Per-dimension MSE / Var:")
        for i in range(actuals.shape[1]):
            dim_mse = mean_squared_error(actuals[:, i], predictions[:, i])
            dim_var = np.var(actuals[:, i])
            ratio = (dim_mse / dim_var) * 100 if dim_var > 0 else float('inf')
            print(f"  Dim {i}: MSE = {dim_mse:.6f}, Var = {dim_var:.6f}, MSE/Var = {ratio:.2f}%")
    # plot_prediction_errors(actual_states, predicted_states)
    # mse = mean_squared_error(actual_states, predicted_states)
    # print(f"Overall MSE: {mse:.6f}")
    #return np.mean(errors) if errors else float('inf')


from sklearn.metrics import mean_squared_error

def evaluate_model_on_testset(model, test_data):
    """
    Evaluates the model on a fixed test set of (state, action, faulty next_state) tuples.

    Returns:
        float: Overall MSE
    """
    actuals = []
    predictions = []

    for state, action, true_next_state in test_data:
        predicted = model.predict(state, action)[0]
        actuals.append(true_next_state)
        predictions.append(predicted)

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    mse = np.mean((actuals - predictions) ** 2)
    var = np.var(actuals)
    print(f"[Test Set] MSE: {mse:.6f}, Variance: {var:.6f}, MSE/Var = {mse / var:.2%}")

    if actuals.shape[1] > 1:
        print("Per-dimension MSE / Var (Test Set):")
        for i in range(actuals.shape[1]):
            dim_mse = mean_squared_error(actuals[:, i], predictions[:, i])
            dim_var = np.var(actuals[:, i])
            ratio = (dim_mse / dim_var) * 100 if dim_var > 0 else float('inf')
            print(f"  Dim {i}: MSE = {dim_mse:.6f}, Var = {dim_var:.6f}, MSE/Var = {ratio:.2f}%")




def evaluate_fault_inference_accuracy(domain_name, model_name, all_fault_modes,models_by_fault, num_tests=20):
    correct = 0
    confusion = defaultdict(int)

    for fault_mode in all_fault_modes:
        print(f"\n⚡ Testing with TRUE fault: {fault_mode}")

        for test_id in range(num_tests):
            # Generate a single trajectory under this fault
            trajectory, exec_len, success = wfe.execute_with_faults(
                domain_name=domain_name,
                debug_print=False,
                execution_fault_mode_name=fault_mode,  # ground truth!
                instance_seed=np.random.randint(1_000_000),
                fault_probability=1.0,  # always fault
                render_mode=None,
                ml_model_name=model_name,
                fault_mode_generator=FaultModeGeneratorDiscrete(),
                max_exec_len=200
            )
            # Load trained policy ONCE before running tests
            policy, _ = load_policy(
                domain_name=domain_name,
                ml_model_name=model_name,
                render_mode=None
            )

            # Extract segment: state6 → state12
            obs_seq, action_seq = separate_trajectory(trajectory)
            if len(obs_seq) < 13:  # too short
                continue

            start_state = obs_seq[6]
            true_final_state = obs_seq[12]

            # Run inference
            predicted_fault, mse, all_mses = infer_fault_from_segment(
                models_by_fault=models_by_fault,
                policy=policy,
                domain_name=domain_name,
                start_state=start_state,
                n_steps=6,
                true_final_state=true_final_state,
                refiners=refiners
            )

            confusion[(fault_mode, predicted_fault)] += 1
            if predicted_fault == fault_mode:
                correct += 1

    total = sum(confusion.values())
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Fault inference accuracy: {accuracy * 100:.2f}% over {total} samples.")
    return confusion




