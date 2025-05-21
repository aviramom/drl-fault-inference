import random

import numpy as np
import gym
from gymnasium.utils.step_api_compatibility import DoneStepType
from stable_baselines3.common.env_util import make_atari_env

from rl_models import models
from state_refiners import refiners
from wrappers import wrappers

def execute_with_faults(domain_name,
                        debug_print,
                        execution_fault_mode_name,
                        instance_seed,
                        fault_probability,
                        render_mode,
                        ml_model_name,
                        fault_mode_generator,
                        max_exec_len):
    """
    Executes one trajectory in a Gym or Atari environment with faults injected.

    Automatically dispatches to either `execute_gym_with_faults` or `execute_atari_with_faults`
    based on the domain.

    Parameters:
        domain_name (str): Gym or Atari environment name (e.g., "CartPole_v1", "Breakout_v4").
        debug_print (bool): Whether to print debugging info for each step.
        execution_fault_mode_name (str): The name of the fault mode to inject.
        instance_seed (int): Random seed for reproducibility.
        fault_probability (float): Probability of injecting a fault on each action step.
        render_mode (str or None): Gym render mode (e.g., "rgb_array", None).
        ml_model_name (str): Name of the RL policy/model to load and execute.
        fault_mode_generator (FaultModeGenerator): Generator for creating fault behavior.
        max_exec_len (int): Maximum steps to run the trajectory.

    Returns:
        tuple:
            - trajectory (list): List of alternating observations and actions.
            - exec_len (int): Number of steps executed.
            - success (bool): Whether the episode reached a success condition (not always applicable).
    """

    if domain_name in ['Breakout_v4', 'Pong_v4']:
        trajectory, exec_len,success  = execute_atari_with_faults(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, render_mode, ml_model_name, fault_mode_generator, max_exec_len)
    else:
        trajectory, exec_len,success  = execute_gym_with_faults(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, render_mode, ml_model_name, fault_mode_generator, max_exec_len)

    return trajectory, exec_len,success

def execute_gym_with_faults(domain_name,
                             debug_print,
                             execution_fault_mode_name,
                             instance_seed,
                             fault_probability,
                             render_mode,
                             ml_model_name,
                             fault_mode_generator,
                             max_exec_len):
    """
    Executes a Gym environment with injected faults on selected actions.

    Parameters:
        domain_name (str): Name of the Gym environment (e.g., "CartPole_v1").
        debug_print (bool): Whether to print each action and observation.
        execution_fault_mode_name (str): Label of the fault mode applied during execution.
        instance_seed (int): Seed to initialize the environment.
        fault_probability (float): Probability of action being faultily altered.
        render_mode (str or None): Rendering mode ("human", "rgb_array", or None).
        ml_model_name (str): Name of the RL model used.
        fault_mode_generator (FaultModeGenerator): Generator for applying fault behavior.
        max_exec_len (int): Maximum number of steps to run.

    Returns:
        tuple:
            - trajectory (list): Alternating sequence of observations and actions.
            - exec_len (int): Total steps taken.
            - success (bool): Whether the episode completed "successfully".
    """
    # initialize environment
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode = fault_mode_generator.generate_fault_mode(execution_fault_mode_name)

    # initializing empty trajectory
    trajectory = []

    action_number = 1
    done = False
    exec_len = 1
    obs, _ = env.reset()
    while not done and exec_len < max_exec_len:
        trajectory.append(obs)
        if debug_print:
            print(f'a#:{action_number} [PREVOBS]: {obs.tolist() if not isinstance(obs, int) else obs}')
        action, _ = model.predict(refiners[domain_name](obs), deterministic=True)
        action = int(action)
        trajectory.append(action)
        if random.random() < fault_probability:
            faulty_action = execution_fault_mode(action)
        else:
            faulty_action = action
        if debug_print:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}')
        obs, reward, done, trunc, info = env.step(faulty_action)
        if debug_print:
            print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
        action_number += 1
        exec_len += 1
    success=done
    if domain_name=='CartPole_v1':
        success = not done
    trajectory.append(obs)
    env.close()

    return trajectory, exec_len,success


def execute_atari_with_faults(domain_name,
                              debug_print,
                              execution_fault_mode_name,
                              instance_seed,
                              fault_probability,
                              render_mode,
                              ml_model_name,
                              fault_mode_generator,
                              max_exec_len):
    """
    Executes an Atari environment (e.g., Breakout or Pong) with injected action faults.

    Parameters:
        domain_name (str): Name of the Atari environment (must end with "_v4").
        debug_print (bool): Print step-wise debug output.
        execution_fault_mode_name (str): Name of the injected fault pattern.
        instance_seed (int): Seed for environment.
        fault_probability (float): Probability of injecting a fault.
        render_mode (str or None): Unused for Atari but passed for API consistency.
        ml_model_name (str): Name of the model to use.
        fault_mode_generator (FaultModeGenerator): Generator to inject fault behavior.
        max_exec_len (int): Maximum number of steps to simulate.

    Returns:
        tuple:
            - trajectory (list): Alternating sequence of observations and actions.
            - exec_len (int): Number of steps executed.
            - success (bool): True if episode finished, False otherwise.
    """
    print(f'executing {domain_name} with fault mode: {execution_fault_mode_name}\n========================================================================================')

    # initialize environment
    env = make_atari_env('Breakout-v4', n_envs=1, seed=instance_seed)
    initial_obs = env.reset()
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode = fault_mode_generator.generate_fault_mode(execution_fault_mode_name)

    # initializing empty trajectory
    trajectory = []


    action_number = 1
    done = False
    exec_len = 1
    obs = env.reset()
    while not done and exec_len < max_exec_len:
        trajectory.append(obs)
        if debug_print:
            print(f'a#:{action_number} [PREVOBS]: {obs.tolist() if not isinstance(obs, int) else obs}')
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])
        trajectory.append(action)
        if random.random() < fault_probability:
            faulty_action = execution_fault_mode(action)
        else:
            faulty_action = action
        if debug_print:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}')
        faulty_action = np.array([faulty_action])
        obs, reward, done, info = env.step(faulty_action)
        if debug_print:
            print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
        action_number += 1
        exec_len += 1
    success=done
    trajectory.append(obs)
    env.close()

    return trajectory, exec_len,success


def collect_gym_with_faults_data(num_of_trajectories,
                                  domain_name,
                                  debug_print,
                                  fault_mode_names,
                                  fault_probability,
                                  render_mode,
                                  ml_model_name,
                                  fault_mode_generator,
                                  max_exec_len):
    """
    Collects multiple Gym trajectories with randomly selected fault modes applied per episode.

    A different fault mode from `fault_mode_names` is selected uniformly at random for each
    trajectory execution.

    Parameters:
        num_of_trajectories (int): Number of trajectories to collect.
        domain_name (str): Gym environment name (e.g., "CartPole_v1").
        debug_print (bool): Enable or disable detailed logging of steps.
        fault_mode_names (list[str]): List of fault mode names to sample from.
        fault_probability (float): Chance of injecting a fault at any given step.
        render_mode (str or None): Rendering mode passed to the environment.
        ml_model_name (str): Name of the RL model used.
        fault_mode_generator (FaultModeGenerator): Responsible for generating fault behavior.
        max_exec_len (int): Maximum number of timesteps per trajectory.

    Returns:
        list of list: A list of trajectories, each consisting of alternating observations and actions.
    """


    # initialize execution fault mode
    #     trajectories = []
    #     for i in range(num_of_tralectorirs):
    #
    #     # initializing empty trajectory
    #         instance_seed = random.randint(0, 1000000)
    #         trajectory, exec_len, success = execute_gym_no_faults(domain_name, debug_print, instance_seed, render_mode, ml_model_name, max_exec_len)
    #         trajectories.append(trajectory)
    #     return trajectories

    # initializing empty trajectory
    trajectories = []
    for i in range(num_of_trajectories):
    # initializing empty trajectory
            instance_seed = random.randint(0, 1000000)
            i = random.randint(0, len(fault_mode_names)-1)
            fault_name = fault_mode_names[i]
            trajectory, exec_len, success = execute_gym_with_faults(domain_name, debug_print,fault_name, instance_seed,fault_probability, render_mode, ml_model_name, fault_mode_generator,max_exec_len)
            trajectories.append(trajectory)
    return trajectories




