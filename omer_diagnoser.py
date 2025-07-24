import numpy as np


def simulate_faulty_run(
    start_state,
    policy,
    env,
    models_by_fault,
    fault_mode_str,
    domain_name,
    num_steps,
    refiners
):
    """
    Simulates a rollout of `num_steps` steps under a given fault mode using a hybrid model+env.

    Args:
        start_state (np.ndarray): Initial state to start simulation from.
        policy (SB3 model): The trained policy.
        env (gym.Env): The environment (should support set_state, step).
        models_by_fault (dict): Nested dict[fault_mode_str][faulty_action] → model
        fault_mode_str (str): Fault mode key, e.g. "[0,1,2,2]"
        domain_name (str): Name of the Gym domain (used for refinement).
        num_steps (int): Number of simulation steps.
        refiners (dict): Maps domain_name → refinement function for policy input.

    Returns:
        List[dict]: Simulated trajectory of transitions.
    """
    fault_mapping = eval(fault_mode_str)
    state = np.array(start_state)
    trajectory = []

    for _ in range(num_steps):
        refined_state = refiners[domain_name](state)
        tried_action, _ = policy.predict(refined_state, deterministic=True)
        tried_action = int(tried_action)

        # Store current state before transition
        current_state = state

        if fault_mode_str in models_by_fault and tried_action in models_by_fault[fault_mode_str]:
            model = models_by_fault[fault_mode_str][tried_action]
            used_model = True
            next_state = model.predict(current_state)
        else:
            used_model = False
            obs, _, _, _, _ = env.step(tried_action)
            next_state = obs

        trajectory.append({
            'state': current_state,
            'tried_action': tried_action,
            'used_model': used_model,
            'next_state': next_state
        })

        state = next_state

    return trajectory
