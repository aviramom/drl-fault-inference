from random import random

import numpy as np
from rollout import *
from omer_diagnoser import *
from state_refiners import refiners
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
from FaultyEnvironment import FaultyEnvironment
from persist_models import *


if __name__ == '__main__':
    # ======= CONFIGURATION ========
    domains_files1 = ['e2000_Acrobot.json','e3000_CartPole.json',
                     'e6000_FrozenLake.json', 'e4000_MountainCar.json', 'e5000_Taxi.json']
    domains_files = ['e2000_Acrobot.json','e3000_CartPole.json','e4000_MountainCar.json',
                     'e5000_Taxi.json','e6000_FrozenLake.json']

    debug_mode = False
    render_mode = 'rgb_array'   # "human", "rgb_array"
    max_exec_len = 400
    num_of_trajectories = 60
    fault_probability = 1  # always inject fault
    fault_mode_generator = FaultModeGeneratorDiscrete()

    metadata_string=''
    #train models for all domains and save them!!!!############
    for domain in domains_files:

        # ======= LOAD PARAMETERS ========
        param_dict = read_json_data(f"experimental inputs/{domain}")
        domain_name = param_dict['domain_name']
        model_name = param_dict['ml_model_name']
        all_fault_modes = param_dict['possible_fault_mode_names']
        ml_model_name = param_dict['ml_model_name']
        model_type='linear'

        models =train_models_for_fault_modes(domain_name,
                                             model_name,
                                             all_fault_modes,
                                             fault_mode_generator,
                                             num_of_trajectories,
                                             debug_mode,
                                             fault_probability,
                                             render_mode,
                                             max_exec_len,
                                             model_type)

        for fm_dict in models.values():
            for model in fm_dict.values():
                metadata_string += f'\n Domain: {domain_name} | {model.get_metadata_string()}'

        save_models_by_fault(models, domain_name,model_name)


    print(metadata_string)

    # for fault_mode, action in models.items():
    #     for faulty_action, model in action.items():
    #         print(f"\n📊 Fault Mode: {model.fault_mode}")
    #
    #         num_dims = model.Y.shape[1]  # number of output dimensions
    #
    #         for dim in range(num_dims):
    #             print(f"📈 Plotting regression lines from input features to output dimension {dim}")
    #             model.print_regression_equation(output_dim=dim)
    #             #model.plot_all_feature_regressions(output_dim=dim)




    #Dictionary to hold faulty envs per fault mode
    # faulty_envs = {}
    # policy, env = load_policy(domain_name, ml_model_name, render_mode)
    #
    # # Build each FaultyEnvironment
    # for fault_mode, model in models.items():
    #     fault_env = FaultyEnvironment(
    #         fault_mode=fault_mode,
    #         fault_model=model,
    #         policy=policy,
    #         env=env,
    #         domain_name= domain_name
    #     )
    #     faulty_envs[str(fault_mode)] = fault_env  # use string as key for clarity
    #     start_state = np.array([0,1,0,-1]) # or one from a trajectory
    #     steps = 5
    #
    #     for fault_mode, env in faulty_envs.items():
    #         print(f"\n🔍 Fault Mode: {fault_mode}")
    #         traj = env.rollout(n_steps=steps, start_state=start_state)
    #         for step in traj:
    #             print(step)



















