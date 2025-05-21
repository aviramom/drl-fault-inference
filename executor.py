import random

import numpy as np
import gym
from stable_baselines3.common.env_util import make_atari_env

from rl_models import models
from state_refiners import refiners
from wrappers import wrappers

def execute(domain_name,
            debug_print,
            execution_fault_mode_name,
            instance_seed,
            fault_probability,
            render_mode,
            ml_model_name,
            fault_mode_generator,
            max_exec_len):
    if domain_name in ['Breakout_v4', 'Pong_v4']:
        trajectory, faulty_actions_indices = execute_atari(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, render_mode, ml_model_name, fault_mode_generator, max_exec_len)
    else:
        trajectory, faulty_actions_indices = execute_gym(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, render_mode, ml_model_name, fault_mode_generator, max_exec_len)

    return trajectory, faulty_actions_indices

def execute_gym(domain_name,
                debug_print,
                execution_fault_mode_name,
                instance_seed,
                fault_probability,
                render_mode,
                ml_model_name,
                fault_mode_generator,
                max_exec_len):
    print(f'executing with fault mode: {execution_fault_mode_name}\n========================================================================================')

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

    faulty_actions_indices = []
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
        if faulty_action != action:
            faulty_actions_indices.append(action_number)
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

    trajectory.append(obs)
    env.close()

    return trajectory, faulty_actions_indices


def execute_atari(domain_name,
                  debug_print,
                  execution_fault_mode_name,
                  instance_seed,
                  fault_probability,
                  render_mode,
                  ml_model_name,
                  fault_mode_generator,
                  max_exec_len):
    print(f'executing with fault mode: {execution_fault_mode_name}\n========================================================================================')

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

    faulty_actions_indices = []
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
        if faulty_action != action:
            faulty_actions_indices.append(action_number)
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

    trajectory.append(obs)
    env.close()

    return trajectory, faulty_actions_indices


def execute_manual(domain_name,
                   debug_print,
                   execution_fault_mode_name,
                   instance_seed,
                   fault_probability,
                   render_mode,
                   ml_model_name,
                   fault_mode_generator,
                   execution_length,
                   faulty_actions_indices,):
    print(f'executing with fault mode: {execution_fault_mode_name}\n========================================================================================')

    # initialize environment
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode = fault_mode_generator.generate_fault_model(execution_fault_mode_name)

    # initializing empty trajectory
    trajectory = []

    action_number = 1
    done = False
    exec_len = 1
    obs, _ = env.reset()
    while not done and action_number < execution_length + 2:
        trajectory.append(obs)
        if debug_print:
            print(f'a#:{action_number} [PREVOBS]: {list(obs)}')
        action, _ = model.predict(refiners[domain_name](obs), deterministic=True)
        action = int(action)
        trajectory.append(action)
        if action_number in faulty_actions_indices:
            faulty_action = execution_fault_mode(action)
        else:
            faulty_action = action
        if debug_print:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}\n')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}\n')
        obs, reward, done, trunc, info = env.step(faulty_action)
        action_number += 1
        exec_len += 1

    env.close()

    return trajectory, faulty_actions_indices
############################################################################################################
#omer's part
###############################################################################################################


####################no faults####################################33
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


################################ with faults ######################################################
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
    if domain_name in ['Breakout_v4', 'Pong_v4']:
        trajectory, exec_Len,success  = execute_atari_with_faults(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, render_mode, ml_model_name, fault_mode_generator, max_exec_len)
    else:
        trajectory, exec_Len,success  = execute_gym_with_faults(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, render_mode, ml_model_name, fault_mode_generator, max_exec_len)

    return trajectory, exec_Len,success

def execute_gym_with_faults(domain_name,
                debug_print,
                execution_fault_mode_name,
                instance_seed,
                fault_probability,
                render_mode,
                ml_model_name,
                fault_mode_generator,
                max_exec_len):
    print(f'executing {domain_name} with fault mode: {execution_fault_mode_name}\n========================================================================================')

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




