from fault_mode_generators import FaultModeGeneratorDiscrete
from pipeline import separate_trajectory, read_json_data
import with_faults_executor as wfe
from random import random
import random
import numpy as np
import gym
from gym.envs.classic_control import MountainCarEnv
from stable_baselines3.common.env_util import make_atari_env

from rl_models import models
from state_refiners import refiners
from wrappers import wrappers


def get_faulty_tuples(domain_name, render_mode, trajectory):
    """
    Identifies and returns faulty (state, action, next_state) transitions in a single trajectory.

    A tuple is considered faulty if executing the action in the recorded pre-state does not
    result in the expected post-state (as recorded in the trajectory). This usually happens
    due to an injected fault that alters the agent's behavior.

    Parameters:
        domain_name (str): Name of the Gym environment (e.g., "CartPole-v1").
        render_mode (str): Rendering mode for the Gym environment (e.g., "rgb_array" or None).
        trajectory (list): A single trajectory containing observations and actions as returned
                           by the simulator, typically a list of (obs, action) pairs.

    Returns:
        list of tuples: A list of faulty transitions (state, action, next_state), where the
                        observed transition did not match the environment's prediction under
                        the intended action.
    """
    faulty_tuples = []
    actions, obs = separate_trajectory(trajectory)
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    env.reset()
    for i in range(len(actions)):
        s, a, s_next = obs[i], actions[i], obs[i + 1]
        t = (s, a, s_next)

        try:
            env.unwrapped.state = s  # attempt to set the internal state directly
        except AttributeError:
            print("Error: Environment does not allow setting state directly.")
            continue  # skip if state cannot be set

        pred_obs, reward, done, trunc, info = env.step(a)

        # compare predicted vs actual next state
        if not np.allclose(pred_obs, s_next, atol=1e-5):
            faulty_tuples.append(t)

    return faulty_tuples





def get_faulty_data(num_of_trajectories, domain_name, debug_mode, fault_name,
                    fault_probability, render_mode, ml_model_name,
                    fault_mode_generator, max_exec_len):
    """
    Generates and collects faulty (state, action, next_state) transitions from multiple trajectories
    using a specified fault injection mode.

    Parameters:
        num_of_trajectories (int): Number of trajectories to collect.
        domain_name (str): Name of the Gym environment (e.g., "CartPole-v1").
        debug_mode (bool): Whether to enable debug output for trajectory generation.
        fault_name (str): Name/label of the fault being injected (e.g., "action_flip").
        fault_probability (float): Probability of injecting a fault at each timestep.
        render_mode (str): Rendering mode for Gym (e.g., None, "rgb_array").
        ml_model_name (str): Name of the policy/model used to run the environment.
        fault_mode_generator (FaultModeGenerator): Object to define which faults are injected.
        max_exec_len (int): Maximum length for each trajectory.

    Returns:
        list of tuples: A flattened list of faulty transitions (state, action, next_state) across all trajectories.
    """
    trajectories = collect_gym_trajectories_with_faults(
        num_of_trajectories, domain_name, debug_mode, fault_name,
        fault_probability, render_mode, ml_model_name,
        fault_mode_generator, max_exec_len
    )

    result = []
    print(f'The number of trajectories: {len(trajectories)}')
    counter = 0

    for t in trajectories:
        faulty_tuples = get_faulty_tuples(domain_name, render_mode, t)
        for tup in faulty_tuples:
            counter += 1
            result.append(tup)

    print(f'Total faulty transitions collected: {counter}')
    return result



def get_augmented_faulty_data(domain_name, fault_mode_name, model_name, fault_mode_generator,
                               num_samples=1000, render_mode=None):
    """
    Generates additional (state, action, faulty_next_state) transitions where a fault actually occurred.
    Only retains samples where the faulty action differs from the intended action.
    """
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    faulty_action_fn = fault_mode_generator.generate_fault_mode(fault_mode_name)
    transitions = []

    for _ in range(num_samples):
        seed = np.random.randint(0, 1_000_000)
        state, _ = env.reset(seed=seed)

        try:
            env.unwrapped.state = state
        except AttributeError:
            continue

        action_space = env.action_space
        intended_action = action_space.sample()
        faulty_action = faulty_action_fn(intended_action)

        # ✅ Keep only if the action was actually faulted
        if intended_action != faulty_action:
            next_state, _, done, trunc, _ = env.step(faulty_action)
            if not (done or trunc):
                transitions.append((state, intended_action, next_state))

    env.close()
    print(f"Augmented faulty transitions collected: {len(transitions)}")
    return transitions




def collect_gym_trajectories_with_faults(num_of_trajectories,
                                         domain_name,
                                         debug_print,
                                         fault_mode_name,
                                         fault_probability,
                                         render_mode,
                                         ml_model_name,
                                         fault_mode_generator,
                                         max_exec_len):
    """
    Collects multiple trajectories from a Gym environment with faults injected according to a specified fault mode.

    This function runs the given environment multiple times using the specified DRL model and fault injection settings.
    Each trajectory consists of (observation, action) pairs, where some actions may be altered based on the fault mode.

    Parameters:
        num_of_trajectories (int): Number of faulty trajectories to generate.
        domain_name (str): The name of the Gym environment (e.g., "CartPole-v1").
        debug_print (bool): Whether to print debug output during execution.
        fault_mode_name (str): Identifier or label for the current fault mode (e.g., "flip_action_0_to_1").
        fault_probability (float): Probability that a fault is injected at each step.
        render_mode (str or None): Rendering mode for Gym (e.g., "rgb_array", "human", or None).
        ml_model_name (str): Name of the policy/model to use (must match your registry or load logic).
        fault_mode_generator (FaultModeGenerator): Object controlling the injected faults per step.
        max_exec_len (int): Maximum number of steps to run per trajectory.

    Returns:
        list: A list of trajectories, where each trajectory is a list of (observation, action) pairs.
              Faults may be applied during generation based on the fault mode and probability.
    """
    print(f"Executing {domain_name} with fault mode: {fault_mode_name}\n"
          f"{'=' * 88}")

    trajectories = []

    for i in range(num_of_trajectories):
        instance_seed = random.randint(0, 1000000)
        trajectory, exec_len, success = wfe.execute_with_faults(
            domain_name,
            debug_print,
            fault_mode_name,
            instance_seed,
            fault_probability,
            render_mode,
            ml_model_name,
            fault_mode_generator,
            max_exec_len
        )
        trajectories.append(trajectory)

    return trajectories



def get_all_transitions_from_trajectory(domain_name, render_mode, trajectory):
    """
    Returns all (state, action, next_state) transitions from a single trajectory,
    regardless of whether a fault actually occurred.

    Parameters:
        domain_name (str): Gym environment name.
        render_mode (str or None): Rendering mode.
        trajectory (list): List of (obs, action) pairs.

    Returns:
        list of (state, action, next_state) transitions.
    """
    all_tuples = []
    actions, obs = separate_trajectory(trajectory)
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    env.reset()

    for i in range(len(actions)):
        s, a, s_next = obs[i], actions[i], obs[i + 1]
        all_tuples.append((s, a, s_next))

    env.close()
    return all_tuples



def get_all_transitions_under_fault(num_of_trajectories, domain_name, debug_mode, fault_name,
                                    fault_probability, render_mode, ml_model_name,
                                    fault_mode_generator, max_exec_len):
    """
    Collects all (state, action, next_state) transitions from trajectories run under a fault mode,
    regardless of whether a fault occurred at each step.

    Parameters: (same as get_faulty_data)

    Returns:
        list of (state, action, next_state) transitions.
    """
    trajectories = collect_gym_trajectories_with_faults(
        num_of_trajectories, domain_name, debug_mode, fault_name,
        fault_probability, render_mode, ml_model_name,
        fault_mode_generator, max_exec_len
    )

    result = []
    print(f'The number of trajectories: {len(trajectories)}')
    counter = 0

    for t in trajectories:
        transitions = get_all_transitions_from_trajectory(domain_name, render_mode, t)
        for tup in transitions:
            counter += 1
            result.append(tup)

    print(f'Total transitions collected under fault mode: {counter}')
    return result
