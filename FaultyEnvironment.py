import numpy as np
import ast

#from Demos.security.get_policy_info import domain_name

from state_refiners import refiners

class FaultyEnvironment:
    def __init__(self, env, policy, fault_mode, fault_model,domain_name):
        """
        Args:
            env: Gym environment instance (must support .reset(), .step())
            policy: SB3 policy model (must support .predict(obs, deterministic=True))
            fault_mode: list mapping each tried action to actual executed action
            fault_model: trained FaultyTransitionModel for this fault mode
        """
        self.env = env
        self.policy = policy
        self.fault_mode = fault_mode
        self.fault_model = fault_model
        self.last_obs = None
        self.domain_name= domain_name

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.last_obs = obs
        return obs

    def set_state(self, state):
        """Directly sets the internal state (e.g., for diagnosis starting point)."""
        self.last_obs = np.array(state)

    def get_state(self):
        """Returns current internal observation."""
        return self.last_obs


    def step(self):
        """
        Steps the environment forward using the fault model if a fault is triggered.

        Returns:
            next_obs: The resulting observation after applying action (real or faulted)
            tried_action: The action intended by the policy
            actual_action: The action executed (same as tried if no fault, else faulted)
            used_model: Boolean flag (True if used fault model, False if real env)
        """
        tried_action, _ = self.policy.predict(refiners[self.domain_name](self.last_obs), deterministic=True)
        tried_action = int(self.fault_mode[tried_action])
        actual_action = self.fault_mode[tried_action]
        used_model = (actual_action != tried_action)

        if used_model:
            # Use the trained faulty model to simulate the result
            predicted_next_obs = self.fault_model.predict(refiners[self.domain_name](self.last_obs), tried_action)
            predicted_next_obs = predicted_next_obs.flatten()
            self.last_obs = predicted_next_obs
        else:
            self.env.set_state(self.last_obs)
            obs, reward, done, trunc, info = self.env.step(actual_action)
            self.last_obs = obs

        return self.last_obs, tried_action, actual_action, used_model



    def rollout(self, n_steps, start_state=None):
        """
        Performs n_steps starting from current state or a given state.

        Args:
            n_steps (int): Number of steps to simulate
            start_state (np.ndarray or None): If provided, sets internal state before starting

        Returns:
            trajectory (list of dict): Each dict has:
                - 'state': current state before action
                - 'tried_action': action proposed by policy
                - 'actual_action': action after fault mode
                - 'next_state': resulting state
                - 'used_model': whether the fault model was used
        """
        if start_state is not None:
            self.set_state(start_state)

        trajectory = []

        for _ in range(n_steps):
            current_state = self.last_obs
            tried_action, _ = self.policy.predict(refiners[self.domain_name](current_state), deterministic=True)
            tried_action = int(tried_action)
            # Ensure fault_mode is a list of ints
            if isinstance(self.fault_mode, str):
                fault_mapping = ast.literal_eval(self.fault_mode)
            else:
                fault_mapping = self.fault_mode

            actual_action = int(fault_mapping[tried_action])
            used_model = (actual_action != tried_action)

            if used_model:
                #next_state = self.fault_model.predict(current_state, tried_action).flatten()
                refined = current_state
                predicted = self.fault_model.predict(refined,tried_action).flatten()
                next_state = predicted[:len(current_state)]  # Ensure shape matches
            else:
                obs = self.env.reset()
                self.env.set_state(self.last_obs)
                obs, reward, done, trunc, info = self.env.env.step(actual_action)
                next_state = obs

            # Update internal state
            self.last_obs = next_state

            # Record transition
            trajectory.append({
                'state': current_state,
                'tried_action': tried_action,
                'actual_action': actual_action,
                'next_state': next_state,
                'used_model': used_model
            })

        return trajectory

