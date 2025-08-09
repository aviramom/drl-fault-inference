import copy
import time

import gym
import numpy as np
import persist_models as pm
from h_consts import DETERMINISTIC
from h_raw_state_comparators import comparators
from rl_models import models
from state_refiners import refiners
from wrappers import wrappers


def W(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    ts0 = time.time()
    b = 0
    e = len(observations) - 1
    S = observations[0]
    for i in range(1, len(observations)):
        a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
        a = int(a)
        S, reward, done, trunc, info = simulator.step(a)
        if observations[i] is not None:
            if comparators[domain_name](observations[i], S):
                b = i
            else:
                e = i
                if debug_print:
                    print(f"i broke at {i}")
                break
    D = []
    for i in range(b + 1, e + 1):
        D.append(i)
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    # finilizing the runtime in ms
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    output = {
        "diagnoses": D,
        "init_rt_sec": 0.0,
        "init_rt_ms": 0.0,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": diagnosis_runtime_sec,
        "totl_rt_ms": diagnosis_runtime_ms,
        "G_max_size":0
    }

    return output


def SN(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        G[key_j] = [candidate_fault_modes[key_j], A_j, S_0]
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # running the diagnosis loop
    ts1 = time.time()
    for i in range(1, len(observations)):
        irrelevant_keys = []
        for key_j in G.keys():
            a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
            a_gag_i_j = G[key_j][0](a_gag_i)
            simulator.set_state(G[key_j][2])
            S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
            G[key_j][1].append(int(a_gag_i_j))
            G[key_j][2] = S_gag_i_j
            if observations[i] is not None:
                if not comparators[domain_name](observations[i], S_gag_i_j):
                    irrelevant_keys.append(key_j)

        # remove the irrelevant fault modes
        for key in irrelevant_keys:
            G.pop(key)

        if debug_print:
            print(f'STEP {i}/{len(observations)}: KICKED {len(irrelevant_keys)} ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break
    te1 = time.time()
    diagnosis_runtime_sec += te1 - ts1

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": len(candidate_fault_modes)
    }

    return raw_output


def fm_and_state_in_set(key_raw, state, FG):
    for fkey in FG.keys():
        fkey_raw = fkey.split('_')[0]
        fstate = FG[fkey][2]
        if key_raw == fkey_raw and state == fstate:
            return True
    return False


def SIF(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], A_j, S_0]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    for i in range(1, len(observations)):
        ts1 = time.time()
        irrelevant_keys = []
        new_relevant_keys = {}
        for key_j in G.keys():
            a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
            a_gag_i = int(a_gag_i)
            a_gag_i_j = G[key_j][0](a_gag_i)

            # apply the normal and the faulty action on the reconstructed states, respectively
            simulator.set_state(G[key_j][2])
            S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
            simulator.set_state(G[key_j][2])
            S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
            if observations[i] is not None:
                # the case where there is an observation that can be checked
                S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i))
                    G[key_j][2] = S_gag_i
                elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i))
                    G[key_j][2] = S_gag_i
                elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    irrelevant_keys.append(key_j)
                elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i_j))
                    G[key_j][2] = S_gag_i_j
            else:
                # the case where there is no observation to be checked - insert the normal action and state to the original key
                if debug_print:
                    print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                G[key_j][1].append(int(a_gag_i))
                G[key_j][2] = S_gag_i
                if a_gag_i != a_gag_i_j:
                    # if the action was changed - create new trajectory and insert it as well
                    if debug_print:
                        print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    A_j_to_fault = copy.deepcopy(G[key_j][1])
                    A_j_to_fault[-1] = a_gag_i_j
                    k_j = key_j.split('_')[0]
                    new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                    I[k_j] = I[k_j] + 1
        # add new relevant fault modes
        for key in new_relevant_keys:
            G[key] = new_relevant_keys[key]
        # remove the irrelevant fault modes
        for key in irrelevant_keys:
            G.pop(key)
        te1 = time.time()
        diagnosis_runtime_sec += te1 - ts1

        # filter out similar trajectories (applies to taxi only)
        if domain_name == "Taxi_v3":
            FG = {}
            for key in G.keys():
                key_raw = key.split('_')[0]
                state = G[key][2]
                if not fm_and_state_in_set(key_raw, state, FG):
                    FG[key] = G[key]
            G = FG

        # update the maximum size of G
        G_max_size = max(G_max_size, len(G))

        if debug_print:
            if observations[i] is not None:
                print(f'STEP {i}/{len(observations)}: OBSERVED')
            else:
                print(f'STEP {i}/{len(observations)}: HIDDEN')
            print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
            print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
            print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = j - i
            i = j
    sorted_index_pairs = sorted(index_pairs.keys(), key=lambda k: (index_pairs[k], k))
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in sorted_index_pairs]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU2(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = j - i
            i = j
    useful_index_pairs = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            useful_index_pairs[pair] = index_pairs[pair]
    sorted_useful_index_pairs = sorted(useful_index_pairs.keys(), key=lambda k: (useful_index_pairs[k], k))
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in sorted_useful_index_pairs]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU3(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    index_pairs_failed = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], set()]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    index_pairs_failed_sorted_useful = {}
    action_types_combined = set()
    for pair in index_pairs_failed_sorted:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b+i}: a {a}')
            index_pairs_failed_sorted[pair][1].add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if len(index_pairs_failed_sorted[pair][1].difference(action_types_combined)) != 0:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            action_types_combined.update(index_pairs_failed_sorted[pair][1])
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU4(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    index_pairs_failed = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], set()]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    index_pairs_failed_sorted_useful = {}
    action_types_combined = set()
    for pair in index_pairs_failed_sorted:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b+i}: a {a}')
            index_pairs_failed_sorted[pair][1].add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if len(index_pairs_failed_sorted[pair][1].difference(action_types_combined)) != 0:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            action_types_combined.update(index_pairs_failed_sorted[pair][1])
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed_sorted_useful.keys():
        actions = index_pairs_failed_sorted_useful[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU5(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    # sort the pairs that failed according to length
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted.keys()]
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed_sorted.keys():
        actions = index_pairs_failed_sorted[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU6(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    # sort the pairs that failed according to length
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed_sorted.keys():
        actions = index_pairs_failed_sorted[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    # filter unuseful failed pairs (this can be done here after we filtered the fault modes)
    index_pairs_failed_sorted_useful = {}
    pairs_unique_action_sets = []
    for pair in index_pairs_failed_sorted:
        if index_pairs_failed_sorted[pair][1] not in pairs_unique_action_sets:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            pairs_unique_action_sets.append(index_pairs_failed_sorted[pair][1])
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU7(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed.keys()]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU8(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed.keys()]
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed.keys():
        actions = index_pairs_failed[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output



# --- You must implement/provide this loader. It should return:
# models_by_fault: dict[fault_mode_str][action] -> model (with .predict(state)->next_state)
def get_models_by_fault(domain_name, ml_model_name):
    return pm.load_models_by_fault(domain_name, ml_model_name)



######## my diagnoser####################
def SIFM(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    """
    Model-based diagnoser:
      - candidate_fault_modes: dict[name -> (ignored function)]. We only use the keys.
      - observations: list of states (some entries may be None).
    Returns same shape as SIF(): dict with 'diagnoses', timings, etc.
    """
    # 1) Load policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # 2) Env as a fallback simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    S0, _ = simulator.reset(seed=instance_seed)
    S0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S0), "First obs must match reset state"

    # 3) Load your trained models
    models_by_fault = get_models_by_fault(domain_name, ml_model_name)

    # timers
    t_init = time.time()

    # 4) Init hypotheses: one per fault mode key
    G = {}
    branch_ids = {key: 0 for key in candidate_fault_modes.keys()}
    for fm_key in candidate_fault_modes.keys():
        G[fm_key + f"_{branch_ids[fm_key]}"] = [fm_key, [], S0]
        branch_ids[fm_key] += 1

    init_sec = time.time() - t_init
    diag_sec = 0.0
    G_max = len(G)

    # 5) Main loop over time
    for i in range(1, len(observations)):
        step_start = time.time()
        obs_i = observations[i]
        to_remove = []
        to_add = {}

        # iterate on a snapshot of keys (we will mutate G)
        for key in list(G.keys()):
            fm_key, acts, S_curr = G[key]

            # policy action
            S_arr = np.array(S_curr[0] if isinstance(S_curr, tuple) else S_curr).flatten()
            a, _ = policy.predict(refiners[domain_name](S_arr), deterministic=DETERMINISTIC)
            a = int(a.item()) if isinstance(a, np.ndarray) else int(a)

            # ENV step from S_curr
            simulator.set_state(S_curr)
            S_env, _, _, _, _ = simulator.step(a)

            # MODEL step if exists
            S_model = None
            if fm_key in models_by_fault and a in models_by_fault[fm_key]:
                S_model = models_by_fault[fm_key][a].predict(S_arr).flatten()

            if obs_i is not None:
                env_ok   = comparators[domain_name](S_env,   obs_i)
                model_ok = comparators[domain_name](S_model, obs_i) if S_model is not None else False

                if env_ok and model_ok:
                    # keep ENV (simplest). Optionally branch model too.
                    if debug_print:
                        print(f"[t={i}][{fm_key}] both match → keep ENV")
                    G[key][1] = acts + [a]
                    G[key][2] = S_env
                    # Optional branch for MODEL:
                    # child = fm_key + f"_{branch_ids[fm_key]}"
                    # to_add[child] = [fm_key, acts + [a], S_model]
                    # branch_ids[fm_key] += 1

                elif env_ok and not model_ok:
                    if debug_print:
                        print(f"[t={i}][{fm_key}] env-only → keep ENV")
                    G[key][1] = acts + [a]
                    G[key][2] = S_env

                elif (not env_ok) and model_ok:
                    if debug_print:
                        print(f"[t={i}][{fm_key}] model-only → keep MODEL")
                    G[key][1] = acts + [a]
                    G[key][2] = S_model

                else:
                    if debug_print:
                        print(f"[t={i}][{fm_key}] neither → prune")
                    to_remove.append(key)

            else:
                # Hidden step: keep ENV; optionally branch MODEL if different
                if debug_print:
                    print(f"[t={i}][{fm_key}] hidden → keep ENV")
                G[key][1] = acts + [a]
                G[key][2] = S_env

                if S_model is not None and not comparators[domain_name](S_model, S_env):
                    child = fm_key + f"_{branch_ids[fm_key]}"
                    if debug_print:
                        print(f"[t={i}][{fm_key}] hidden → branch MODEL child {child}")
                    to_add[child] = [fm_key, acts + [a], S_model]
                    branch_ids[fm_key] += 1

        # apply adds/removes
        for k in to_remove:
            if k in G:
                del G[k]
        for k, v in to_add.items():
            G[k] = v

        diag_sec += time.time() - step_start
        G_max = max(G_max, len(G))
        if debug_print:
            print(f"STEP {i}: added {len(to_add)} | pruned {len(to_remove)} | alive {len(G)}\n")
        if len(G) == 1:
            if debug_print:
                print(f"Early stop: single hypothesis at t={i}")
            break

    # timings
    return {
        "diagnoses": G,
        "init_rt_sec": init_sec,
        "init_rt_ms": init_sec * 1000,
        "diag_rt_sec": diag_sec,
        "diag_rt_ms": diag_sec * 1000,
        "totl_rt_sec": init_sec + diag_sec,
        "totl_rt_ms": (init_sec + diag_sec) * 1000,
        "G_max_size": G_max
    }



diagnosers = {
    # new fault models
    "W": W,
    "SN": SN,
    "SIF": SIF,
    "SIFU": SIFU,
    "SIFU2": SIFU2,
    "SIFU3": SIFU3,
    "SIFU4": SIFU4,
    "SIFU5": SIFU5,
    "SIFU6": SIFU6,
    "SIFU7": SIFU7,
    "SIFU8": SIFU8,
    "SIFM": SIFM
}
