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
    state = start_state
    trajectory = []

    for _ in range(num_steps):

        state = np.array(state).flatten()


        refined_state = refiners[domain_name](state)
        tried_action, _ = policy.predict(refined_state, deterministic=True)
        tried_action = int(tried_action.item()) if isinstance(tried_action, np.ndarray) else int(tried_action)

        # Store current state before transition
        current_state = state

        if fault_mode_str in models_by_fault and tried_action in models_by_fault[fault_mode_str]:
            model = models_by_fault[fault_mode_str][tried_action]
            used_model = True
            next_state = model.predict(current_state).flatten()
        else:
            used_model = False
            env.set_state(current_state)
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



import time, copy
import numpy as np
import gym
from h_consts import DETERMINISTIC
from h_raw_state_comparators import comparators
from rl_models import models
from state_refiners import refiners
from wrappers import wrappers

# expects these globals like your SIF() had
# models: dict of SB3 classes by name
# wrappers: dict mapping domain_name -> env wrapper class
# refiners: dict mapping domain_name -> fn(state)->policy_obs
# comparators: dict mapping domain_name -> fn(sim_state, obs)->bool
DETERMINISTIC = True

def diagnose_with_models(
    debug_print,
    render_mode,
    instance_seed,
    ml_model_name,
    domain_name,
    observations,
    candidate_fault_modes,   # iterable of fault_mode_str keys you want to test
    models_by_fault,         # dict[fault_mode_str][action] -> FaultyTransitionModel
    use_refiner_for_model=False  # set True only if your models were trained on refined inputs
):
    """
    Multiple-hypothesis diagnoser using (fault_mode, action) models.

    Hypothesis = (fault_mode_str, action_sequence_so_far, reconstructed_state)
    At each step:
      - Get policy action 'a' from reconstructed state
      - Compute S_env by stepping the real env with 'a'
      - If a model exists for (fault_mode_str, a): S_model = model.predict(state)
      - Compare each to the observation (if provided):
          * If only S_env matches -> keep hypothesis with S_env
          * If only S_model matches -> keep hypothesis with S_model
          * If both match -> keep S_env (and optionally branch a model child if you want)
          * If neither match -> prune hypothesis
      - If observation is None:
          * Keep the S_env path
          * If model exists and S_model != S_env -> branch a second hypothesis for S_model

    Returns:
      {
        "diagnoses": G,  # dict[key] = [fault_mode_str, actions_list, reconstructed_state]
        "init_rt_sec": ..., "diag_rt_sec": ..., "totl_rt_sec": ...,
        "init_rt_ms": ..., "diag_rt_ms": ..., "totl_rt_ms": ...,
        "G_max_size": int
      }
    """
    # 1) Load policy (no env bound here; we’ll create env below as simulator)
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # 2) Simulator env (real env path when we don’t have a model)
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0), "First obs must match reset state"

    # timers
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0
    G_max_size = 0

    # unique ids per fault mode key for branching
    I = {fm_key: 0 for fm_key in candidate_fault_modes}

    # 3) Initialize hypotheses G
    ts0 = time.time()
    G = {}
    for fm_key in candidate_fault_modes:
        G[fm_key + f'_{I[fm_key]}'] = [fm_key, [], S_0]   # [fault_mode_str, actions, state]
        I[fm_key] += 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # 4) Step through time
    for i in range(1, len(observations)):
        ts1 = time.time()
        irrelevant_keys = []
        new_relevant_keys = {}

        for key in list(G.keys()):
            fm_key, A, S_curr = G[key]

            # --- policy action from reconstructed state
            # ensure correct shape for policy input
            S_arr = np.array(S_curr)
            if isinstance(S_curr, tuple):
                S_arr = np.array(S_curr[0])
            S_arr = S_arr.flatten()

            policy_in = refiners[domain_name](S_arr)
            a, _ = policy.predict(policy_in, deterministic=DETERMINISTIC)
            a = int(a.item()) if isinstance(a, np.ndarray) else int(a)

            # --- simulate normal env step
            simulator.set_state(S_curr)
            S_env, _, _, _, _ = simulator.step(a)

            # --- simulate "faulted" via model if available
            has_model = (fm_key in models_by_fault) and (a in models_by_fault[fm_key])
            if has_model:
                model_input = refiners[domain_name](S_arr) if use_refiner_for_model else S_arr
                S_model = models_by_fault[fm_key][a].predict(model_input).flatten()
            else:
                S_model = None

            obs_i = observations[i]

            if obs_i is not None:
                # compare to observation
                env_match   = comparators[domain_name](S_env,   obs_i)
                model_match = comparators[domain_name](S_model, obs_i) if S_model is not None else False

                if env_match and model_match:
                    # ambiguous: both paths match
                    if debug_print:
                        print(f"t={i} [{fm_key}] case both-match: keep ENV (a={a})")
                    A.append(a)
                    G[key][1] = A
                    G[key][2] = S_env
                    # optional branching: keep a model child too
                    # child_key = fm_key + f'_{I[fm_key]}'
                    # new_relevant_keys[child_key] = [fm_key, A[:-1] + [a], S_model]
                    # I[fm_key] += 1

                elif env_match and not model_match:
                    if debug_print:
                        print(f"t={i} [{fm_key}] case env-only: keep ENV (a={a})")
                    A.append(a)
                    G[key][1] = A
                    G[key][2] = S_env

                elif not env_match and model_match:
                    if debug_print:
                        print(f"t={i} [{fm_key}] case model-only: keep MODEL (a={a})")
                    A.append(a)  # action attempted by policy
                    G[key][1] = A
                    G[key][2] = S_model

                else:
                    if debug_print:
                        print(f"t={i} [{fm_key}] case neither: prune")
                    irrelevant_keys.append(key)

            else:
                # hidden step: keep ENV path; optionally branch a MODEL child if it differs
                if debug_print:
                    print(f"t={i} [{fm_key}] hidden: keep ENV (a={a})")
                A_env = A + [a]
                G[key][1] = A_env
                G[key][2] = S_env

                if has_model:
                    # branch only if model outcome is meaningfully different
                    different = not comparators[domain_name](S_model, S_env)
                    if different:
                        child_key = fm_key + f'_{I[fm_key]}'
                        if debug_print:
                            print(f"t={i} [{fm_key}] hidden: branch MODEL child (a={a}) -> {child_key}")
                        new_relevant_keys[child_key] = [fm_key, A + [a], S_model]
                        I[fm_key] += 1

        # apply branching & pruning
        for k in new_relevant_keys:
            G[k] = new_relevant_keys[k]
        for k in irrelevant_keys:
            if k in G:
                G.pop(k)

        te1 = time.time()
        diagnosis_runtime_sec += te1 - ts1

        # update max width
        G_max_size = max(G_max_size, len(G))

        if debug_print:
            added = list(new_relevant_keys.keys())
            print(f"STEP {i}/{len(observations)-1}: added {len(added)} | kicked {len(irrelevant_keys)} | alive {len(G)}")
            if len(added):
                print("  added:", added)
            print("  alive keys:", list(G.keys()))
            print()

        # stop early if single hypothesis remains
        if len(G) == 1:
            if debug_print:
                print(f"Early stop at step {i}: single hypothesis left.")
            break

    # finalize timings
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    return {
        "diagnoses": G,  # key -> [fault_mode_str, actions_list, reconstructed_state]
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }
