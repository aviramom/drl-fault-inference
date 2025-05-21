import numpy as np

import no_faults_executor as nfe
import with_faults_executor as wfe
from FaultyTransitionModel import FaultyTransitionModel
from evaluation import evaluate_model_on_faults, evaluate_fault_inference_accuracy
from evaluation import evaluate_model_on_testset
from fault_mode_generators import FaultModeGeneratorDiscrete
from pipeline import read_json_data
from pipeline import separate_trajectory
from Faulty_Data_Extractor import get_faulty_data, get_augmented_faulty_data, get_all_transitions_under_fault
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # ======= CONFIGURATION ========
    domains_files = ['Acrobot.json', 'Breakout.json', 'CartPole.json',
                     'FrozenLake.json', 'MountainCar.json', 'Taxi.json']

    debug_mode = False
    render_mode = 'rgb_array'
    max_exec_len = 200
    num_of_trajectories = 60
    domain = domains_files[2]  # e.g., Acrobot
    fault_probability = 100  # always inject fault
    fault_mode_generator = FaultModeGeneratorDiscrete()

    # ======= LOAD PARAMETERS ========
    param_dict = read_json_data(f"omer's experimental inputs/{domain}")
    domain_name = param_dict['domain_name']
    model_name = param_dict['ml_model_name']
    all_fault_modes = param_dict['all_fault_names']

    models_by_fault = {}

    # ======= TRAIN A MODEL PER FAULT MODE ========
    for fault_mode in all_fault_modes:
        print(f"\n==== Training model for fault mode: {fault_mode} ====")

        # GET ALL TRANSITIONS under fault mode
        trajectory_data = get_all_transitions_under_fault(
            num_of_trajectories,
            domain_name,
            debug_mode,
            fault_mode,
            fault_probability,
            render_mode,
            model_name,
            fault_mode_generator,
            max_exec_len
        )

        print(f"Total training samples (all transitions under fault): {len(trajectory_data)}")

        # TRAIN / TEST SPLIT
        train_data, test_data = train_test_split(trajectory_data, test_size=0.2, random_state=42)
        print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")

        # TRAIN MODEL
        model = FaultyTransitionModel(
            fault_mode=fault_mode,
            data=train_data
        )

        # SAVE MODEL BY FAULT NAME
        models_by_fault[fault_mode] = model

        # EVALUATE ON TEST SET
        evaluate_model_on_testset(model, test_data)

        # EVALUATE GENERALIZATION
        evaluate_model_on_faults(
            model=model,
            domain_name=domain_name,
            fault_mode_name=fault_mode,
            fault_mode_generator=fault_mode_generator,
            ml_model_name=model_name,
            num_samples=2000,
            render_mode=render_mode
        )

    print("\n✅ All models trained and evaluated.")
    # ======= INFERENCE EVALUATION ========
    print("\n🔍 Starting fault inference evaluation...")
    # confusion = evaluate_fault_inference_accuracy(
    #     domain_name=domain_name,
    #     model_name=model_name,
    #     all_fault_modes=all_fault_modes,
    #     models_by_fault=models_by_fault,
    #     num_tests=20  # You can increase this for more robustness
    # )




















    # domains_files=['Acrobot.json', 'Breakout.json','CartPole.json','FrozenLake.json', 'MountainCar.json','Taxi.json']
    # print(f'running all domains with no faults\n=====================================')
    # debug_mode=False
    # render_mode='rgb_array'
    # max_exec_len=200
    # domain=domains_files[0]
    # fault_mode_generator = FaultModeGeneratorDiscrete()
    # param_dict = read_json_data(f"omer's experimental inputs/{domain}")
    # trajectories= nfe.collect_gym_no_faults_data(10,param_dict['domain_name'],debug_mode,render_mode,param_dict['ml_model_name'],max_exec_len)
    # print("free fault trajectories####################")
    # for t in trajectories:
    #     actions, obs = separate_trajectory(t)
    #     print(actions)
    # print("with fault trajectories####################")
    # trajectories = wfe.collect_gym_with_faults_data(10, param_dict['domain_name'], debug_mode,param_dict['all_fault_names'],param_dict['fault_probability'], render_mode,
    #
    #                                               param_dict['ml_model_name'],fault_mode_generator, max_exec_len)
    # for t in trajectories:
    #     actions, obs = separate_trajectory(t)
    #     print(actions)
    # good_domains_no_faults= []
    # bad_domains_no_faults=[]
    # good_domains_with_faults = []
    # bad_domains_with_faults = []
    # fault_mode_generator = FaultModeGeneratorDiscrete()
    # for domain in domains_files:
    #     #no faults
    #     param_dict = read_json_data(f"omer's experimental inputs/{domain}")
    #     trajectory, exec_len, success = nfe.execute_no_faults(param_dict['domain_name'],debug_mode,1,render_mode,param_dict['ml_model_name'],max_exec_len)
    #     if success:
    #         good_domains_no_faults.append(param_dict['domain_name'])
    #     else:
    #         bad_domains_no_faults.append(param_dict['domain_name'])
    #     print('=================================')
    #     print('domain:', param_dict['domain_name'])
    #     print("execution length:", exec_len)
    #     print("succes:", success)
    #     #with faults
    #     trajectory, exec_len, success = wfe.execute_with_faults(param_dict['domain_name'], debug_mode,
    #                                                             param_dict['fault_mode_name'], 1,
    #                                                             param_dict['fault_probability'], render_mode,
    #                                                             param_dict['ml_model_name'], fault_mode_generator,
    #                                                             max_exec_len)
    #     if success:
    #         good_domains_with_faults.append(param_dict['domain_name'])
    #     else:
    #         bad_domains_with_faults.append(param_dict['domain_name'])
    #     print('=================================')
    #     print('domain:', param_dict['domain_name'])
    #     print("execution length:", exec_len)
    #     print("succes:", success)
    #     print('=================================')
    #     print('=================================')
    # print('=======================conclussion=================================')
    # print("good domains no faults:", good_domains_no_faults)
    # print("bad domains no faults:", bad_domains_no_faults)
    # print('=====================================================')
    # print("good domains with faults:", good_domains_with_faults)
    # print("bad domains with faults:", bad_domains_with_faults)
