import numpy as np
from rollout import *
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

if __name__ == '__main__':
    # ======= CONFIGURATION ========
    domains_files = ['e2000_Acrobot.json', 'e7000_Breakout.json', 'e3000_CartPole.json',
                     'e6000_FrozenLake.json', 'e4000_MountainCar.json', 'e5000_Taxi.json']

    debug_mode = False
    render_mode = 'rgb_array'
    max_exec_len = 200
    num_of_trajectories = 60
    domain = domains_files[2]  # e.g., Acrobot
    fault_probability = 100  # always inject fault
    fault_mode_generator = FaultModeGeneratorDiscrete()

    # ======= LOAD PARAMETERS ========
    param_dict = read_json_data(f"experimental inputs/{domain}")
    domain_name = param_dict['domain_name']
    model_name = param_dict['ml_model_name']
    all_fault_modes = param_dict['possible_fault_mode_names']
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
    for fault_mode, model in models.items():
        print(f"\n📊 Fault Mode: {fault_mode}")

        num_dims = model.Y.shape[1]  # number of output dimensions

        for dim in range(num_dims):
            print(f"📈 Plotting regression lines from input features to output dimension {dim}")
            model.print_regression_equation(output_dim=dim)
            model.plot_all_feature_regressions(output_dim=dim)

















