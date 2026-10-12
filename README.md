# drl-fault-inference

Model-based diagnosis of actuator faults in deep-RL agents: learn per-fault transition models from
faulty rollouts, then use them to infer which fault mode explains an observed trajectory.

## Problem

A trained deep-RL policy (Stable-Baselines3 PPO / DQN) acts in a Gym environment. At execution
time the agent may be **faulty**: the action it chooses is not the action that gets executed.

- A **fault mode** is a deterministic action remapping, written as a list. For a 3-action domain,
  `[0,2,1]` means "action 0 stays 0, action 1 becomes 2, action 2 becomes 1".
  `[0,0,0]` means every action collapses to action 0.
- With **fault probability** `p`, each step's chosen action is replaced by its remapped action
  (`executor.py`). The indices of the steps where the executed action actually changed are recorded
  as ground truth.
- The diagnoser sees only the policy, the sequence of observed states (optionally with some states
  hidden), and a set of **candidate fault modes**. It does not see which actions were executed.

**What is inferred:** which candidate fault mode(s) are consistent with the observed trajectory,
ranked, together with a reconstruction of the action sequence that each surviving hypothesis
implies.

## Method

### Pipeline (`main.py` -> `pipeline.run_experimental_setup_new`)

For each experiment file in `experimental inputs/`:

1. **Execute.** Load the trained policy, run it in the environment for up to 50 steps while
   injecting the true fault mode with probability `p`, and record the trajectory
   (`executor.execute`). Runs with no faulty step are re-sampled.
2. **Observe.** Split the trajectory into actions and states, then mask states according to
   `percent_visible_states`; the first and last states are always visible.
3. **Candidates.** Build the candidate set: the true fault mode plus `num_candidate_fault_modes - 1`
   random others, shuffled.
4. **Diagnose.** Run each configured diagnoser (`p_diagnosers.diagnosers`).
5. **Rank.** `rank_diagnoses_SFM` scores each surviving hypothesis by the fraction of steps where
   the fault mode *could* have changed the action and the reconstructed action differs from the
   policy's action.
6. **Record.** Write per-run details to `experimental results/<exp>.xlsx` and a comparison workbook
   (`runs`, `runs_flat`, `summary_by_diagnoser`, `acc_by_domain` sheets) to `<results dir>/<exp>__compare.xlsx`.

### Diagnosers

`p_diagnosers.py` contains a family of consistency-based diagnosers that keep a set `G` of
hypotheses (fault mode + reconstructed actions + current state), step them forward, prune the
ones that contradict an observation, and branch when a state is hidden:

| Name | Transition source | Notes |
|------|-------------------|-------|
| `W`, `SN`, `SIF`, `SIFU`…`SIFU8` | The real environment, stepped through `wrappers.py` with the candidate fault mapping applied explicitly | Simulator-based variants. `SIFU5` is the baseline used in the reported experiments. |
| `SIFM` | The real environment for the nominal step **plus a learned regression model** for the faulty step | The learned-model diagnoser. It needs no explicit fault function, only the trained models. |
| `SIFU8M`, `SIFM_scored` | Variants of the above | Experimental; not used in the reported runs. |

### Regression models as faulty transition models

`train_linear_models.py` / `train_non_linear_models.py` (via `rollout.train_models_for_fault_modes`):

1. For each fault mode, collect 60 trajectories (max 400 steps) with the fault always active
   (`p = 1`), using `Faulty_Data_Extractor.collect_trajectories_with_faults`.
2. Split **by trajectory** 80/20 to avoid train/test leakage, then extract `(s, a, s')` tuples.
3. For each action that the fault mode actually remaps, fit one model `s -> s'` on the tuples where
   the policy chose that action:
   - **linear**: `FaultyTransitionModel` (scikit-learn `LinearRegression`);
   - **polynomial**: `PolyDeltaModel`, a `StandardScaler -> PolynomialFeatures(degree) -> Ridge(alpha=3)`
     pipeline that predicts the state delta `s' - s` (degree 2 or 3).
4. Save as `trained_*_models/<Domain>/<Algo>/<fault_mode>/model_action_<a>.joblib` plus a
   `metadata.json` (`persist_models.py`).

Inside `SIFM`, at each step the policy picks action `a` from the hypothesis' current state. The
environment gives the nominal next state, and `models_by_fault[fault_mode][a]` (if it exists)
predicts the faulty next state. If a state is visible, a hypothesis survives when either prediction
matches it within `atol = rtol = 1e-2`. If the state is hidden, the hypothesis follows the nominal
branch and, when the model disagrees, spawns a second branch that follows the model.

## Environments

| Experiment file | Gym environment | Policy | Candidate fault modes |
|-----------------|-----------------|--------|-----------------------|
| `e2000_Acrobot.json` | `Acrobot-v1` | PPO | 10 |
| `e3000_CartPole.json` | `CartPole-v1` | PPO | 3 |
| `e4000_MountainCar.json` | `MountainCar-v0` | DQN | 10 |
| `e5000_Taxi.json` | `Taxi-v3` | PPO | 10 |
| `e6000_FrozenLake.json` | `FrozenLake-v1` | PPO | 10 |
| `e7000_Breakout.json` | `Breakout-v4` (Atari) | A2C | not run (no diagnoser configured) |

`wrappers.py` adds `set_state` to each environment so that diagnosers can restart simulation from an
arbitrary state. `state_refiners.py` converts raw simulator state into the policy's observation
format (for example `cos`/`sin` features for Acrobot).

## Results

The numbers below are copied from the `summary_by_diagnoser` sheet of the `*__compare.xlsx`
workbooks in this repo. The experimental setting is the same everywhere: fault probability in
{0.2, 0.4, 0.6, 0.8, 1.0}, one instance seed, **100% of states visible**, and every candidate fault
mode offered. That gives 50 runs per diagnoser per domain (15 for CartPole, which has 3 fault modes).

**Metric caveat.** The sheet column is named `top1_acc`, but the code computes it as *"the true
fault mode is in the diagnoser's final hypothesis set"* (`contains_true`), not as top-1 accuracy
after ranking. "Avg. #hyp" is the average size of that final set. A value below 1 means some runs
ended with every hypothesis pruned.

True fault in final set (avg. #hypotheses):

| Domain | SIFU5 (simulator baseline) | SIFM + linear | SIFM + poly deg 2 | SIFM + poly deg 3 |
|--------|---------------------------|---------------|-------------------|-------------------|
| Acrobot-v1 | 1.00 (2.5–2.6) | 0.48 (0.74) | 0.48 (0.66) | 0.46 (0.74) |
| CartPole-v1 | 1.00 (1.87) | 1.00 (1.47) | 0.87 (1.53) | 0.93 (1.67) |
| MountainCar-v0 | 1.00 (2.64) | 0.04 (0.36) | 0.22 (1.34) | 0.04 (0.36) |
| Taxi-v3 | 1.00 (1.64) | 0.42 (1.02) | 0.26 (0.90) | 0.42 (1.02) |
| FrozenLake-v1 | 0.34 (1.22) | 0.06 (0.84) | 0.06 (0.84) | 0.06 (0.84) |

Sources: `liner regression models experimental results/`,
`non-linear 2-degree regression models experimental results/`,
`non-linear 3-degree regression models experimental results/`. `SIFU5` does not use the learned
models. Its inclusion rate is the same in all three folders, and its avg. #hyp differs only on
Acrobot (2.54 / 2.48 / 2.60).
Average wall-clock time per diagnosis (`avg_total_ms` in the same sheets) ranges from about 6 ms
(FrozenLake) to about 110 ms (`SIFM` on MountainCar).

`experimental results/*__compare.xlsx` holds an additional run (including hand-made pivot sheets).
It is not labelled with the model type it used, so it is not included in the table.

**Still to be added:** a consolidated summary across seeds, partial observability
(`percent_visible_states` < 100), true top-1 accuracy after ranking, and per-fault-probability
breakdowns. The `runs` / `runs_flat` sheets already contain the per-run data needed for these.

## How to run

### Dependencies

```bash
pip install -r requirements.txt
```

The code imports the legacy `gym` package with the 0.26-style API (`reset` returns `(obs, info)`,
`step` returns 5 values), together with `stable_baselines3` and `gymnasium`. Package versions
were not recorded, so `requirements.txt` is unpinned. Atari (Breakout) also needs the ALE ROMs
(`ale-py`).

### Required assets not in this repo

The trained RL policies are loaded from
`environments/<Domain>/models/<Algo>/<Domain>__<Algo>.zip`
(for example `environments/CartPole_v1/models/PPO/CartPole_v1__PPO.zip`). This folder is not tracked
(`*.zip` is git-ignored), so you need to train or provide these policies first.

`main.py` also plays a local `alarm.mp3` through `pygame` when a run finishes. That file is not
tracked, so either supply any mp3 under that name or remove those lines.

### Steps

```bash
# 1. Train faulty-transition models (writes trained_linear_models/ or trained_non_linear_3-degree_models/)
python train_linear_models.py
python train_non_linear_models.py      # degree is PolyDeltaModel(degree=3) by default

# 2. Run the diagnosis experiments for Acrobot, CartPole, MountainCar, Taxi, FrozenLake
python main.py
```

Settings that are currently hard-coded and need to be edited to switch model sets:

- which trained models `SIFM` loads: `get_models_by_fault` in `p_diagnosers.py` (`save_dir=...`);
- where the comparison workbook is written: `run_experimental_setup_new` in `pipeline.py`;
- the experiment grid (fault probabilities, seeds, visibility, diagnosers): the JSON files in
  `experimental inputs/`.

## Repository layout

| Path | Description |
|------|-------------|
| `main.py` | Entry point. Runs the experiment pipeline over the five domains. |
| `pipeline.py` | Experiment loop: execute, mask, build candidates, diagnose, rank, write Excel. |
| `p_diagnosers.py` | All diagnosers (`W`, `SN`, `SIF`, `SIFU*`, `SIFM`, …) and the `diagnosers` registry. |
| `executor.py` | Runs a policy with an injected fault mode (Gym and Atari), plus no-fault variants. |
| `with_faults_executor.py`, `no_faults_executor.py` | Trajectory collection with and without faults, used for training data. |
| `Faulty_Data_Extractor.py` | Turns trajectories into `(s, a, s')` transition tuples. |
| `rollout.py` | `train_models_for_fault_modes` (per-fault, per-action model training) and rollout helpers. |
| `FaultyTransitionModel.py` | Linear / MLP transition model with plotting helpers. |
| `PolyDeltaModel.py` | Polynomial ridge model that predicts state deltas. |
| `FaultyEnvironment.py` | Environment-like wrapper that steps with a learned fault model. |
| `omer_diagnoser.py` | Earlier prototype of model-based simulation and diagnosis. |
| `evaluation.py` | Transition-model evaluation (MSE on faulty transitions / test set). |
| `persist_models.py` | Save and load trained models as `joblib` + `metadata.json`. |
| `train_linear_models.py`, `train_non_linear_models.py` | Scripts that train and save models for all domains. |
| `fault_mode_generators.py` | Turns fault-mode strings into action-mapping functions (discrete and box). |
| `wrappers.py` | Gym wrappers that expose `set_state` / raw state per domain. |
| `state_refiners.py`, `h_raw_state_comparators.py` | Per-domain raw-state to observation conversion and state equality. |
| `rl_models.py`, `h_consts.py`, `utils.py` | SB3 algorithm registry, constants, small helpers. |
| `experimental inputs/` | Experiment grids, one JSON per domain. |
| `omer's experimental inputs/` | Single-run configs used during development. |
| `trained_linear_models/`, `trained_non_linear_2-degree_models/`, `trained_non_linear_3-degree_models/` | Saved faulty-transition models per domain / policy / fault mode / action. |
| `experimental results/` | Per-run Excel output and an additional comparison run. |
| `liner regression models experimental results/`, `non-linear {2,3}-degree regression models experimental results/` | Comparison workbooks (`SIFM` vs `SIFU5`) for each model type. |
