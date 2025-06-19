import copy
import math
import random
import time
from datetime import datetime

import xlsxwriter

from h_common import read_json_data
from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import diagnosers, SIF, SN, W, SIFU, SIFU2, SIFU3, SIFU4, SIFU5, SIFU6, SIFU7, SIFU8
from p_executor import execute


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


def single_experiment_prepare_inputs(domain_name,
                                     ml_model_name,
                                     render_mode,
                                     max_exec_len,
                                     debug_print,
                                     execution_fault_mode_name,
                                     instance_seed,
                                     fault_probability):
    # ### initialize fault model generator
    fault_mode_generator = FaultModelGeneratorDiscrete()

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

    return fault_mode_generator, trajectory_execution, faulty_actions_indices, registered_actions, observations


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
        fm = fault_mode_generator.generate_fault_model(fmr)
        fault_modes[fmr] = fm

    l = list(fault_modes.items())
    random.shuffle(l)
    fault_modes = dict(l)
    return fault_modes


def rank_diagnoses_WFM(raw_output, registered_actions, debug_print):
    # TODO think about more sophisticated ranking elgorithm
    ranking_start_time = time.time()
    G = raw_output['diagnoses']
    # diagnoses = []
    # ranks = []

    diagnoses = G
    ranks = [0] * len(diagnoses)

    ranking_end_time = time.time()
    ranking_runtime_sec = ranking_end_time - ranking_start_time
    ranking_runtime_ms = ranking_runtime_sec * 1000

    output = {
        "diagnoses": diagnoses,
        "init_rt_sec": raw_output["init_rt_sec"],
        "init_rt_ms": raw_output["init_rt_ms"],
        "diag_rt_sec": raw_output["diag_rt_sec"],
        "diag_rt_ms": raw_output["diag_rt_ms"],
        "totl_rt_sec": raw_output["totl_rt_sec"],
        "totl_rt_ms": raw_output["totl_rt_ms"],
        "G_max_size": raw_output['G_max_size'],
        "diagnosis_actions": [],
        "ranks": ranks,
        "rank_rt_sec": ranking_runtime_sec,
        "rank_rt_ms": ranking_runtime_ms,
        "cmpl_rt_sec": raw_output["totl_rt_sec"] + ranking_runtime_sec,
        "cmpl_rt_ms": raw_output["totl_rt_ms"] + ranking_runtime_ms
    }
    return output


def rank_diagnoses_SFM(raw_output, registered_actions, debug_print):
    ranking_start_time = time.time()
    G = raw_output['diagnoses']
    diagnoses = []
    diagnosis_actions = []
    ranks = []

    for key_j in G:
        actions_j = G[key_j][1]
        num_actual_faults = 0
        for i in range(len(actions_j)):
            if actions_j[i] is not None:
                if registered_actions[i] != actions_j[i]:
                    num_actual_faults += 1
        num_potential_faults = 0
        for i in range(len(actions_j)):
            if actions_j[i] is not None:
                a = registered_actions[i]
                fa = G[key_j][0](a)
                if a != fa:
                    num_potential_faults += 1

        if debug_print:
            print(f'num_actual_faults / num_potential_faults: {num_actual_faults} / {num_potential_faults}')

        if num_potential_faults == 0:
            rank = 1.0
        else:
            rank = num_actual_faults * 1.0 / num_potential_faults

        k_j = key_j.split('_')[0]
        if k_j not in diagnoses:
            diagnoses.append(k_j)
            diagnosis_actions.append(actions_j)
            ranks.append(rank)

    zipped_lists = zip(diagnoses, diagnosis_actions, ranks)
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[-1], reverse=True)
    diagnoses, diagnosis_actions, ranks = zip(*sorted_zipped_lists)

    ranking_end_time = time.time()
    ranking_runtime_sec = ranking_end_time - ranking_start_time
    ranking_runtime_ms = ranking_runtime_sec * 1000

    output = {
        "diagnoses": diagnoses,
        "init_rt_sec": raw_output["init_rt_sec"],
        "init_rt_ms": raw_output["init_rt_ms"],
        "diag_rt_sec": raw_output["diag_rt_sec"],
        "diag_rt_ms": raw_output["diag_rt_ms"],
        "totl_rt_sec": raw_output["totl_rt_sec"],
        "totl_rt_ms": raw_output["totl_rt_ms"],
        "G_max_size": raw_output['G_max_size'],
        "diagnosis_actions": diagnosis_actions,
        "ranks": ranks,
        "rank_rt_sec": ranking_runtime_sec,
        "rank_rt_ms": ranking_runtime_ms,
        "cmpl_rt_sec": raw_output["totl_rt_sec"] + ranking_runtime_sec,
        "cmpl_rt_ms": raw_output["totl_rt_ms"] + ranking_runtime_ms
    }

    return output


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


def get_ordinal_rank(diagnoses, ranks, d):
    index_d = diagnoses.index(d)
    rank_d = ranks[index_d]

    unique_ranks_set = set(ranks)
    unique_ranks_list = sorted(list(unique_ranks_set), reverse=True)

    res = unique_ranks_list.index(rank_d)
    return res


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
        {'header': '16_O_candidate_fault_modes'},
        {'header': '17_i_diagnoser'},
        {'header': '18_O_diagnoses'},
        {'header': '19_O_ranks'},
        {'header': '20_O_num_diagnoses'},
        {'header': '21_O_correct_diagnosis_rank'},
        {'header': '22_O_init_rt_sec'},
        {'header': '23_O_init_rt_ms'},
        {'header': '24_O_diag_rt_sec'},
        {'header': '25_O_diag_rt_ms'},
        {'header': '26_O_totl_rt_sec'},
        {'header': '27_O_totl_rt_ms'},
        {'header': '28_O_rank_rt_sec'},
        {'header': '29_O_rank_rt_ms'},
        {'header': '30_O_cmpl_rt_sec'},
        {'header': '31_O_cmpl_rt_ms'},
        {'header': '32_O_G_max_size'},
        {'header': '33_O_diagnosis_actions'}
    ]
    rows = []
    for i in range(len(records)):
        record_i = records[i]
        row = [
            record_i['domain_name'],  # 01_i_domain_name
            record_i['execution_fault_mode_name'],  # 02_i_execution_fault_mode_name
            float(record_i['fault_probability']),  # 03_i_fault_probability
            record_i['instance_seed'],  # 04_i_instance_seed
            str(record_i['faulty_actions_indices']),  # 05_O_faulty_actions_indices
            len(record_i['faulty_actions_indices']),  # 06_O_len_faulty_actions_indices
            str(record_i['registered_actions']),  # 07_O_registered_actions
            len(record_i['registered_actions']),  # 08_O_len_registered_actions
            str(record_i['observations']) if record_i['debug_print'] else 'Omitted',  # 09_O_observations
            record_i['percent_visible_states'],  # 10_i_percent_visible_states
            str(record_i['observation_mask']),  # 11_O_observation_mask
            len(record_i['observation_mask']),  # 12_O_num_visible_states
            record_i['longest_hidden_state_sequence'],  # 13_O_longest_hidden_state_sequence
            str(record_i['masked_observations']) if record_i['diagnoser'] in {"SIF"} else 'Omitted',  # 14_O_masked_observations
            record_i['num_candidate_fault_modes'],  # 15_i_num_candidate_fault_modes
            str(record_i['candidate_fault_modes']),  # 16_O_candidate_fault_modes
            record_i['diagnoser'],  # 17_i_diagnoser
            str(list(record_i['output']['diagnoses'])),  # 18_O_diagnoses
            str(list(record_i['output']['ranks'])),  # 19_O_ranks
            len(record_i['output']['diagnoses']),  # 20_O_num_diagnoses
            get_ordinal_rank(list(record_i['output']['diagnoses']), list(record_i['output']['ranks']), record_i['execution_fault_mode_name']) if record_i['diagnoser'] in {"SN", "SIF", "SIFU", "SIFU2", "SIFU3", "SIFU4", "SIFU5", "SIFU6", "SIFU7", "SIFU8"} else "Irrelevant",  # 21_O_correct_diagnosis_rank
            record_i['output']['init_rt_sec'],  # 22_O_init_rt_sec
            record_i['output']['init_rt_ms'],  # 23_O_init_rt_ms
            record_i['output']['diag_rt_sec'],  # 24_O_diag_rt_sec
            record_i['output']['diag_rt_ms'],  # 25_O_diag_rt_ms
            record_i['output']['totl_rt_sec'],  # 26_O_totl_rt_sec
            record_i['output']['totl_rt_ms'],  # 27_O_totl_rt_ms
            record_i['output']['rank_rt_sec'],  # 28_O_rank_rt_sec
            record_i['output']['rank_rt_ms'],  # 29_O_rank_rt_ms
            record_i['output']['cmpl_rt_sec'],  # 30_O_cmpl_rt_sec
            record_i['output']['cmpl_rt_ms'],  # 31_O_cmpl_rt_ms
            record_i['output']['G_max_size'] if record_i['diagnoser'] in {"SIF", "SIFU", "SIFU2", "SIFU3", "SIFU4", "SIFU5", "SIFU6", "SIFU7", "SIFU8"} else "Irrelevant",  # 32_O_G_max_size
            str(list(record_i['output']['diagnosis_actions'])) if record_i['diagnoser'] in {"SIF"} else "Irrelevant"  # 33_O_diagnosis_actions
        ]
        rows.append(row)
    workbook = xlsxwriter.Workbook(f"experimental results/{experimental_filename.replace('/', '_')}.xlsx")
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(rows), len(columns) - 1, {'data': rows, 'columns': columns})
    workbook.close()


def run_W_single_experiment(domain_name,
                            ml_model_name,
                            render_mode,
                            max_exec_len,
                            debug_print,
                            execution_fault_mode_name,
                            instance_seed,
                            fault_probability,
                            percent_visible_states,
                            possible_fault_mode_names,
                            num_candidate_fault_modes
                            ):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run W
    raw_output = W(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, [], 0,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            {}, output, "W", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_W")

    return raw_output["diag_rt_ms"]


def run_SN_single_experiment(domain_name,
                             ml_model_name,
                             render_mode,
                             max_exec_len,
                             debug_print,
                             execution_fault_mode_name,
                             instance_seed,
                             fault_probability,
                             percent_visible_states,
                             possible_fault_mode_names,
                             num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SN
    raw_output = SN(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SN", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SN")

    return raw_output["diag_rt_ms"]


def run_SIF_single_experiment(domain_name,
                              ml_model_name,
                              render_mode,
                              max_exec_len,
                              debug_print,
                              execution_fault_mode_name,
                              instance_seed,
                              fault_probability,
                              percent_visible_states,
                              possible_fault_mode_names,
                              num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIF(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIF", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIF")

    return raw_output["diag_rt_ms"]


def run_SIFU_single_experiment(domain_name,
                               ml_model_name,
                               render_mode,
                               max_exec_len,
                               debug_print,
                               execution_fault_mode_name,
                               instance_seed,
                               fault_probability,
                               percent_visible_states,
                               possible_fault_mode_names,
                               num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIFU
    raw_output = SIFU(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU")

    return raw_output["diag_rt_ms"]


def run_SIFU2_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU2(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU2", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU2")

    return raw_output["diag_rt_ms"]


def run_SIFU3_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU3(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU3", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU3")

    return raw_output["diag_rt_ms"]


def run_SIFU4_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU4(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU4", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU4")

    return raw_output["diag_rt_ms"]


def run_SIFU5_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU5(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU5", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU5")

    return raw_output["diag_rt_ms"]


def run_SIFU6_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU6(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU6", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU6")

    return raw_output["diag_rt_ms"]


def run_SIFU7_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU7(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU7", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU7")

    return raw_output["diag_rt_ms"]


def run_SIFU8_single_experiment(domain_name,
                                ml_model_name,
                                render_mode,
                                max_exec_len,
                                debug_print,
                                execution_fault_mode_name,
                                instance_seed,
                                fault_probability,
                                percent_visible_states,
                                possible_fault_mode_names,
                                num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    # ### calculate largest hidden gap
    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIFU8(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIFU8", longest_hidden_state_sequence)
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIFU8")

    return raw_output["diag_rt_ms"]


def run_experimental_setup(arguments, render_mode, debug_print):
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

    # ### run the experimental loop
    current_instance_number = 1
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
                        # logging
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        print(f"{dt_string}: {current_instance_number}/{len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'])}")
                        print(f"execution_fault_mode_name: {execution_fault_mode_name}, fault_probability: {fault_probability}, instance_seed: {instance_seed}, percent_visible_states: {percent_visible_states}, num_candidate_fault_modes: {num_candidate_fault_modes}")
                        print(f"{param_dict['diagnoser_name']}\n")

                        # ### prepare candidate fault modes
                        candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, param_dict['possible_fault_mode_names'], fault_mode_generator)

                        # ### run the algorithm
                        diagnoser = diagnosers[param_dict["diagnoser_name"]]
                        raw_output = diagnoser(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

                        # ### ranking the diagnoses
                        if param_dict["diagnoser_name"] == "W":
                            output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)
                        else:
                            output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

                        # ### preparing record for writing to excel file
                        record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, param_dict['possible_fault_mode_names'], num_candidate_fault_modes,
                                                render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                                                candidate_fault_modes, output, param_dict["diagnoser_name"], longest_hidden_state_sequence)
                        records.append(record)
                        current_instance_number += 1

    # ### write records to an excel file
    write_records_to_excel(records, experimental_file_name.split(".")[0])

    print(9)


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
    W_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'][:1]))
    SN_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities'][-1:]) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIF_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU2_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU3_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU4_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU5_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU6_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU7_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))
    SIFU8_instance_number = (len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states'][5:]) * len(param_dict['num_candidate_fault_modes'][1:]))

    total_instances_number = W_instance_number + SN_instance_number + SIF_instance_number + SIFU_instance_number + SIFU2_instance_number + SIFU3_instance_number + SIFU4_instance_number + SIFU5_instance_number + SIFU6_instance_number + SIFU7_instance_number + SIFU8_instance_number
    current_instance_number = 1

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
                            print(f"{dt_string}: {current_instance_number}/{total_instances_number}")
                            print(f"execution_fault_mode_name: {execution_fault_mode_name}, fault_probability: {fault_probability}, instance_seed: {instance_seed}, percent_visible_states: {percent_visible_states}, num_candidate_fault_modes: {num_candidate_fault_modes}, diagnose_name: {diagnoser_name}")
                            print(f"{diagnoser_name}")

                            # ### run the algorithm - except special cases of W, SN
                            if diagnoser_name == "W" and num_candidate_fault_modes != 0:
                                print(f'SKIP\n')
                                continue
                            if diagnoser_name == "SN" and fault_probability != 1.0:
                                print(f'SKIP\n')
                                continue
                            if diagnoser_name != "W" and num_candidate_fault_modes == 0:
                                print(f'SKIP\n')
                                continue
                            if diagnoser_name in ["SIF", "SIFU", "SIFU2"] and percent_visible_states < 30:
                                print(f'SKIP\n')
                                continue
                            if diagnoser_name in ["SIFU3", "SIFU4", "SIFU5", "SIFU6", "SIFU7", "SIFU8"] and percent_visible_states < 30:
                                print(f'SKIP\n')
                                continue
                            diagnoser = diagnosers[diagnoser_name]
                            raw_output = diagnoser(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

                            # ### ranking the diagnoses
                            if diagnoser_name == "W":
                                output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)
                            else:
                                output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

                            # ### preparing record for writing to excel file
                            record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, param_dict['possible_fault_mode_names'], num_candidate_fault_modes,
                                                    render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                                                    candidate_fault_modes, output, diagnoser_name, longest_hidden_state_sequence)
                            records.append(record)

                            print(f'\n')
                            current_instance_number += 1

    # ### write records to an excel file
    write_records_to_excel(records, experimental_file_name.split(".")[0])

    print(9)
