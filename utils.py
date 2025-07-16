from rl_models import models
from wrappers import wrappers
import gym


def load_policy(domain_name, ml_model_name, render_mode):
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)
    return model, env


# separating trajectory to actions and states
def separate_trajectory(trajectory_execution):
    registered_actions = []
    observations = []
    for i in range(len(trajectory_execution)):
        if i % 2 == 1:
            registered_actions.append(trajectory_execution[i])
        else:
            observations.append(trajectory_execution[i])
    if len(registered_actions) == len(observations):
        registered_actions = registered_actions[:-1]
    return registered_actions, observations



def filter_only_faulted_tuples(tuples, fault_mode_str):
    """
    Filters (state, action, next_state) tuples where the fault actually changed the action.

    Args:
        tuples: list of (state, action, next_state)
        fault_mode_str: string representation of the fault mapping, e.g., "[0,0,2,3,4,5]"

    Returns:
        list of only faulty tuples (where action != faulty_action)
    """
    fault_mapping = eval(fault_mode_str)
    return [t for t in tuples if fault_mapping[t[1]] != t[1]]


def filter_only_action_tuples(tuples, action):
    """
    Filters (state, action, next_state) tuples where the fault actually changed the action.

    Args:
        tuples: list of (state, action, next_state)
        action: string representation of the action

    Returns:
        list of only faulty tuples (where action != faulty_action)
    """

    return [t for t in tuples if action == t[1]]
