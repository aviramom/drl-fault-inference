from random import random
import random
import numpy as np
import gym
from gym.envs.classic_control import MountainCarEnv
from stable_baselines3.common.env_util import make_atari_env

from rl_models import models
from state_refiners import refiners
from wrappers import wrappers


def execute_gym_no_faults(domain_name,
                debug_print,
                instance_seed,
                render_mode,
                ml_model_name,
                max_exec_len):
    #print(f'executing with fault mode: {execution_fault_mode_name}\n========================================================================================')

    # initialize environment
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)


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
        if debug_print:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {action}')
        obs, reward, done, trunc, info = env.step(action)
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

def execute_atari_no_faults(domain_name,
                  debug_print,
                  instance_seed,
                  render_mode,
                  ml_model_name,
                  max_exec_len):
    # initialize environment
    env = make_atari_env('Breakout-v4', n_envs=1, seed=instance_seed)
    initial_obs = env.reset()
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)


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

        if debug_print:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {action}')

        action = np.array([action])
        obs, reward, done, info = env.step(action)
        if debug_print:
            print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
        action_number += 1
        exec_len += 1
    success = done
    trajectory.append(obs)
    env.close()

    return trajectory, exec_len, success


def execute_no_faults(domain_name,
            debug_print,
            instance_seed,
            render_mode,
            ml_model_name,
            max_exec_len):
    if domain_name in ['Breakout_v4', 'Pong_v4']:
        trajectory, exec_Len,success = execute_atari_no_faults(domain_name, debug_print, instance_seed, render_mode, ml_model_name, max_exec_len)
    else:
        trajectory, exec_Len,success = execute_gym_no_faults(domain_name, debug_print, instance_seed, render_mode, ml_model_name, max_exec_len)

    return trajectory, exec_Len,success



def execute_gym_no_faults_for_trajectories(domain_name,
                debug_print,
                instance_seed,
                render_mode,
                ml_model_name,
                max_exec_len):
    #print(f'executing with fault mode: {execution_fault_mode_name}\n========================================================================================')

    # initialize environment
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    trajectories = []
    for i in range(2000):

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
            action, _ = model.predict(refiners[domain_name](obs), deterministic=False)
            action = int(action)
            trajectory.append(action)
            if debug_print:
                    print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {action}')
            obs, reward, done, trunc, info = env.step(action)
            if debug_print:
                print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
            action_number += 1
            exec_len += 1
        trajectory.append(obs)
        env.close()
        trajectories.append(trajectory)
    return trajectories



###########################################################################################
def collect_gym_no_faults_data(num_of_tralectorirs,domain_name,
                debug_print,
                render_mode,
                ml_model_name,
                max_exec_len):
    trajectories = []
    for i in range(num_of_tralectorirs):

    # initializing empty trajectory
        instance_seed = random.randint(0, 1000000)
        trajectory, exec_len, success = execute_gym_no_faults(domain_name, debug_print, instance_seed, render_mode, ml_model_name, max_exec_len)
        trajectories.append(trajectory)
    return trajectories
