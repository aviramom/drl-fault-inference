import copy
import math
import random
import json
from datetime import datetime

import numpy as np
import xlsxwriter

from fault_mode_generators import FaultModeGeneratorDiscrete
from executor import execute


def read_json_data(params_file):
    with open(params_file, 'r') as file:
        json_data = json.load(file)
    return json_data

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

def single_experiment_prepare_inputs(domain_name,
                                     ml_model_name,
                                     render_mode,
                                     max_exec_len,
                                     debug_print,
                                     execution_fault_mode_name,
                                     instance_seed,
                                     fault_probability):
    # ### initialize fault model generator
    fault_mode_generator = FaultModeGeneratorDiscrete()

    # ### execute to get trajectory
    trajectory_execution = []
    faulty_actions_indices = []
    while len(faulty_actions_indices) == 0:
        trajectory_execution, faulty_actions_indices = execute(domain_name,
                                                               debug_print,
                                                               execution_fault_mode_name,
                                                               instance_seed,
                                                               fault_probability,
                                                               render_mode,
                                                               ml_model_name,
                                                               fault_mode_generator,
                                                               max_exec_len)

    # ### separating trajectory to actions and states
    registered_actions, observations = separate_trajectory(trajectory_execution)

    # ### make sure the last faulty action was not an action that was removed
    if faulty_actions_indices[-1] > len(registered_actions):
        faulty_actions_indices = faulty_actions_indices[:-1]

    return fault_mode_generator, trajectory_execution, faulty_actions_indices, registered_actions, observations

def generate_observation_mask(observations_length, percent_visible_states):
    mask = [0] * observations_length
    ones = math.floor((observations_length - 2) * percent_visible_states / 100.0)
    indices = random.sample(range(1, observations_length - 1), ones)
    for i in indices:
        mask[i] = 1
    mask[0] = 1
    mask[len(mask) - 1] = 1
    observation_mask = []
    for i in range(len(mask)):
        if mask[i] == 1:
            observation_mask.append(i)
    return observation_mask

def calculate_largest_hidden_gap(observation_mask):
    largest_hidden_gap = 0
    for i in range(1, len(observation_mask)):
        largest_hidden_gap = max(largest_hidden_gap, observation_mask[i] - observation_mask[i - 1] - 1)
    return largest_hidden_gap

def mask_states(observations, observation_mask):
    masked_observations = [None] * len(observations)
    for i in observation_mask:
        masked_observations[i] = copy.deepcopy(observations[i])
    return masked_observations

def prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator):
    if num_candidate_fault_modes == 0:
        candidate_fault_mode_names = []
    else:
        candidate_fault_mode_names = [execution_fault_mode_name]
        rest = copy.deepcopy(possible_fault_mode_names)
        rest.remove(execution_fault_mode_name)
        i = 0
        while i < num_candidate_fault_modes - 1:
            fmr = random.choice(rest)
            candidate_fault_mode_names.append(fmr)
            rest.remove(fmr)
            i += 1
    fault_modes = {}
    for fmr in candidate_fault_mode_names:
        fm = fault_mode_generator.generate_fault_mode(fmr)
        fault_modes[fmr] = fm

    l = list(fault_modes.items())
    random.shuffle(l)
    fault_modes = dict(l)
    return fault_modes

def prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                   render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                   candidate_fault_modes, output, diagnoser, longest_hidden_state_sequence):
    record = {
        "domain_name": domain_name,
        "debug_print": debug_print,
        "execution_fault_mode_name": execution_fault_mode_name,
        "instance_seed": instance_seed,
        "fault_probability": fault_probability,
        "percent_visible_states": percent_visible_states,
        "possible_fault_mode_names": possible_fault_mode_names,
        "num_candidate_fault_modes": num_candidate_fault_modes,
        "render_mode": render_mode,
        "ml_model_name": ml_model_name,
        "max_exec_len": max_exec_len,
        "trajectory_execution": trajectory_execution,
        "faulty_actions_indices": faulty_actions_indices,
        "registered_actions": registered_actions,
        "observations": observations,
        "observation_mask": observation_mask,
        "masked_observations": masked_observations,
        "candidate_fault_modes": list(candidate_fault_modes.keys()),
        "output": output,
        "diagnoser": diagnoser,
        "longest_hidden_state_sequence": longest_hidden_state_sequence
    }
    return record

def write_records_to_excel(records, experimental_filename):
    columns = [
        {'header': '01_i_domain_name'},
        {'header': '02_i_execution_fault_mode_name'},
        {'header': '03_i_fault_probability'},
        {'header': '04_i_instance_seed'},
        {'header': '05_O_faulty_actions_indices'},
        {'header': '06_O_num_faulty_actions'},
        {'header': '07_O_registered_actions'},
        {'header': '08_O_execution_length'},
        {'header': '09_O_observations'},
        {'header': '10_i_percent_visible_states'},
        {'header': '11_O_observation_mask'},
        {'header': '12_O_num_visible_states'},
        {'header': '13_O_longest_hidden_state_sequence'},
        {'header': '14_O_masked_observations'},
        {'header': '15_i_num_candidate_fault_modes'},
        {'header': '16_O_candidate_fault_modes'}
    ]
    rows = []
    for i in range(len(records)):
        record_i = records[i]
        np.set_printoptions(threshold=np.inf)
        row = [
            record_i['domain_name'],  # 01_i_domain_name
            record_i['execution_fault_mode_name'],  # 02_i_execution_fault_mode_name
            float(record_i['fault_probability']),  # 03_i_fault_probability
            record_i['instance_seed'],  # 04_i_instance_seed
            str(record_i['faulty_actions_indices']),  # 05_O_faulty_actions_indices
            len(record_i['faulty_actions_indices']),  # 06_O_len_faulty_actions_indices
            str(record_i['registered_actions']),  # 07_O_registered_actions
            len(record_i['registered_actions']),  # 08_O_len_registered_actions
            str([str(obs) for obs in record_i['observations']]),  # if record_i['debug_print'] else 'Omitted',  # 09_O_observations
            record_i['percent_visible_states'],  # 10_i_percent_visible_states
            str(record_i['observation_mask']),  # 11_O_observation_mask
            len(record_i['observation_mask']),  # 12_O_num_visible_states
            record_i['longest_hidden_state_sequence'],  # 13_O_longest_hidden_state_sequence
            str(record_i['masked_observations']) if record_i['debug_print'] else 'Omitted',  # 14_O_masked_observations
            record_i['num_candidate_fault_modes'],  # 15_i_num_candidate_fault_modes
            str(record_i['candidate_fault_modes'])  # 16_O_candidate_fault_modes
        ]
        rows.append(row)
    workbook = xlsxwriter.Workbook(f"experimental results/{experimental_filename.replace('/', '_')}.xlsx")
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(rows), len(columns) - 1, {'data': rows, 'columns': columns})
    workbook.close()

def run_experimental_setup_new(arguments, render_mode, debug_print):
    # ### parameters dictionary
    experimental_file_name = arguments[1]
    param_dict = read_json_data(f"experimental inputs/{experimental_file_name}")

    # ### prepare the records database to be written to the excel file
    records = []

    # ### the domain name of this experiment (each experiment file has only one associated domain)
    domain_name = param_dict['domain_name']

    # ### the machine learning model name of this experiment (each experiment file has one associated ml model)
    ml_model_name = param_dict['ml_model_name']

    # ### maximum length of the execution for the experiment (each experiment file has one associated length)
    max_exec_len = 200

    # ### preparing index for experimental execution tracing
    # W_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'][:1]))
    # SN_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities'][-1:]) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'][1:]))
    # SIF_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    # SIFU5_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    # SIFSD_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))

    # total_instances_number = W_instance_number + SN_instance_number + SIF_instance_number + SIFU5_instance_number + SIFSD_instance_number
    total_instances_number = 50
    current_instance_number = 1
    start_time = datetime.now()

    # ### run the experimental loop
    for execution_fault_mode_name_i, execution_fault_mode_name in enumerate(param_dict['possible_fault_mode_names']):
        for fault_probability_i, fault_probability in enumerate(param_dict['fault_probabilities']):
            for instance_seed_i, instance_seed in enumerate(param_dict['instance_seeds']):
                # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for ranking
                fault_mode_generator, trajectory_execution, \
                    faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                                ml_model_name,
                                                                                                                render_mode,
                                                                                                                max_exec_len,
                                                                                                                debug_print,
                                                                                                                execution_fault_mode_name,
                                                                                                                instance_seed,
                                                                                                                fault_probability)
                for percent_visible_states_i, percent_visible_states in enumerate(param_dict['percent_visible_states']):
                    # ### generate observation mask
                    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
                    # ### calculate largest hidden gap
                    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
                    print(f'PERCENT VISIBLE STATES: {str(percent_visible_states)}')
                    print(f'OBSERVATION MASK: {str(observation_mask)}')
                    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
                    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
                    print(f'observed {len(observation_mask)}/{len(observations)} states')

                    # ### mask the states list
                    masked_observations = mask_states(observations, observation_mask)

                    for num_candidate_fault_modes_i, num_candidate_fault_modes in enumerate(param_dict['num_candidate_fault_modes']):
                        # ### prepare candidate fault modes
                        candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, param_dict['possible_fault_mode_names'], fault_mode_generator)

                        for diagnoser_name_i, diagnoser_name in enumerate(param_dict['diagnoser_names']):
                            # ### logging
                            now = datetime.now()
                            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                            elapsed_time = now - start_time
                            hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                            minutes, seconds = divmod(remainder, 60)
                            print(f"{dt_string}: {current_instance_number}/{total_instances_number}")
                            print(f"elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                            print(f"execution_fault_mode_name: {execution_fault_mode_name}, fault_probability: {fault_probability}, instance_seed: {instance_seed}, percent_visible_states: {percent_visible_states}, num_candidate_fault_modes: {num_candidate_fault_modes}, diagnose_name: {diagnoser_name}")
                            print(f"{diagnoser_name}")

                            # # ### run the algorithm - except special cases of W, SN
                            # if diagnoser_name == "W" and num_candidate_fault_modes != 0:
                            #     print(f'SKIP\n')
                            #     continue
                            # if diagnoser_name == "SN" and fault_probability != 1.0:
                            #     print(f'SKIP\n')
                            #     continue
                            # if diagnoser_name != "W" and num_candidate_fault_modes == 0:
                            #     print(f'SKIP\n')
                            #     continue
                            # if diagnoser_name in ["SIF"] and percent_visible_states < 30:
                            #     print(f'SKIP\n')
                            #     continue
                            # if diagnoser_name in ["SIFU5"] and percent_visible_states < 30:
                            #     print(f'SKIP\n')
                            #     continue
                            # if diagnoser_name in ["SIFSD"] and percent_visible_states < 30:
                            #     print(f'SKIP\n')
                            #     continue
                            # diagnoser = diagnosers[diagnoser_name]
                            # raw_output = diagnoser(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)
                            #
                            # # ### ranking the diagnoses
                            # if diagnoser_name == "W":
                            #     output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)
                            # elif diagnoser_name == "SIFSD":
                            #     output = rank_diagnoses_SIFSD(raw_output, registered_actions, debug_print)
                            # else:
                            #     output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

                            # ### preparing record for writing to excel file
                            # record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, param_dict['possible_fault_mode_names'], num_candidate_fault_modes,
                            #                         render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            #                         candidate_fault_modes, output, diagnoser_name, longest_hidden_state_sequence)
                            record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, param_dict['possible_fault_mode_names'], num_candidate_fault_modes,
                                                    render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                                                    candidate_fault_modes, None, diagnoser_name, longest_hidden_state_sequence)
                            records.append(record)

                            print(f'\n')
                            current_instance_number += 1

    # ### write records to an excel file
    write_records_to_excel(records, experimental_file_name.split(".")[0])

    print(9)