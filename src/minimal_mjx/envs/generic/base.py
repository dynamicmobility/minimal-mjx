import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from pathlib import Path
from ml_collections import config_dict

import mujoco as mj
from mujoco_playground._src.dm_control_suite import common
from mujoco import mjx
from mujoco_playground._src import mjx_env
from minimal_mjx.utils import EnvState

class SwappableBase(mjx_env.MjxEnv):
    """Atalante walker task."""

    def __init__(self,
            xml_path: Path,
            env_params: config_dict.ConfigDict,
            backend = 'jnp',
            num_free = 3
        ):
        """Initializes the base "swappable" MJX environment for RL training and 
        evaluation. This class implements core functionalities that are common 
        among various roboics tasks. Moreover, this class 
        implements a swappable backend for JAX and NumPy, allowing for 
        flexibility when it comes to training and evaluation (i.e. JaX for
        training and NumPy for evaluation).
        
        Args:
            xml_path: Path to the XML file defining the robot and environment.
            config: Configuration dictionary for the environment.
            config_overrides: Optional dictionary to override specific config values.
            backend: Backend to use for computations, either 'jnp' for JAX or 'np' for NumPy.
        """
        super().__init__(env_params)
        self.params = env_params
        self._xml_path = xml_path.as_posix()
        self._mj_model = mj.MjModel.from_xml_path(
            self._xml_path, common.get_assets()
        )
        self._mj_model.opt.timestep = self.params.sim_dt
        self._mj_model.opt.timestep = self.sim_dt

        self.setup_swappable_backend(backend)
        
        self.nq = self._mj_model.nq
        self.nv = self._mj_model.nv
        self.nu = self._mj_model.nu
        self.qpos_free = num_free
        self.qvel_free = num_free - 1 if num_free == 7 else num_free
        
        self._jt_lims = self._mj_model.jnt_range[num_free:].T
                
    def setup_swappable_backend(self, backend: str):
        """Sets up the backend for the environment."""
        if backend == 'jnp':
            # Setup JAX backend
            self._np = jnp
            self._mj = mjx
            self._uniform = jax.random.uniform
            self._normal = jax.random.normal
            self._bernoulli = jax.random.bernoulli
            self._split = jax.random.split
            self._splice = jax.lax.dynamic_slice
            self._cond = lambda cond, true_fn, false_fn, operand: jax.lax.cond(
                cond, true_fn, false_fn, operand
            )
            self._mjx_model = mjx.put_model(self._mj_model)

            self._step_fn = lambda data, ctrl, model=self._mjx_model: mjx_env.step(
                model, data, ctrl, self.n_substeps
            )

            def mjx_data_init_fn(qpos, qvel, ctrl, time, xfrc_applied):
                data = mjx_env.init(
                    self._mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl
                ).replace(time=time, xfrc_applied=xfrc_applied)
                data = mjx.forward(self._mjx_model, data)
                return data.replace(ctrl=ctrl)

            self._data_init_fn = mjx_data_init_fn

            self._state_init_fn = lambda data, obs, reward, done, metrics, info: mjx_env.State(
                data, obs, reward, done, metrics, info
            )

            self._set_val_fn = lambda arr, val, min_idx=None, max_idx=None: arr.at[min_idx:max_idx].set(val)
            
            self._set_xfrc_fn = lambda data, xfrc_applied: data.replace(xfrc_applied=xfrc_applied)

            self._set_model_params_fn = lambda model, **kwargs: model.replace(
                **kwargs
            )

        elif backend == 'np':
            # Setup numpy backend
            self._np = np
            self._mj = mj
            self._uniform = lambda key, shape=None, minval=0.0, maxval=1.0: np.random.uniform(
                low=minval, high=maxval, size=shape
            )
            self._normal = lambda key, shape=None, loc=0.0, scale=1.0: np.random.normal(
                size=shape
            )
            self._mjx_model = self._mj_model
            self._split = lambda key, num=2: (None for _ in range(num))  # No RNG in np backend
            
            def splice(operand, start_indicies, slice_sizes):
                slices = tuple(slice(start, start + size) for start, size in zip(start_indicies, slice_sizes))
                return operand[slices]
            self._splice = splice

            def cond(cond, true_fn, false_fn, operand):
                if cond:
                    return true_fn(operand)
                else:
                    return false_fn(operand)
            self._cond = cond

            def init_data(model, time, qpos, qvel, ctrl, xfrc_applied):
                data = mj.MjData(model)
                data.time = time
                data.qpos = qpos
                data.qvel = qvel
                data.ctrl = ctrl
                data.xfrc_applied = xfrc_applied
                mj.mj_forward(self.mj_model, data)
                data.ctrl = ctrl

                return data
            self._data_init_fn = lambda time, qpos, qvel, ctrl, xfrc_applied: init_data(
                self._mj_model, time, qpos, qvel, ctrl, xfrc_applied
            )

            def mj_step(model, old_data, ctrl, n_substeps):
                new_data = mj.MjData(model)

                # Allocate state buffer
                state_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
                state = np.empty(state_size)

                # Copy state from old_data
                mj.mj_getState(model, old_data, state, mj.mjtState.mjSTATE_FULLPHYSICS)

                # Set state in new_data
                mj.mj_setState(model, new_data, state, mj.mjtState.mjSTATE_FULLPHYSICS)

                # Step simulation
                for _ in range(n_substeps):
                    new_data.ctrl[:] = ctrl
                    mj.mj_step(model, new_data)

                return new_data
            self._step_fn = lambda data, ctrl, model=self.mjx_model, n_substeps=self.n_substeps: mj_step(
                model, data, ctrl, n_substeps
            )
            
            self._state_init_fn = lambda data, obs, reward, done, metrics, info: EnvState(
                data, obs, reward, done, metrics, info
            )
            def set_val(arr, val, min_idx=None, max_idx=None):
                copy_arr = arr.copy()
                copy_arr[min_idx:max_idx] = val
                return copy_arr
            self._set_val_fn = set_val

            def set_xfrc(data, xfrc_applied):
                data.xfrc_applied = xfrc_applied
                return data
            self._set_xfrc_fn = set_xfrc

            def set_model_params(mj_model, **kwargs):
                for key in kwargs:
                    setattr(mj_model, key, kwargs[key])                

                return mj_model

            self._set_model_params_fn = lambda model, **kwargs: set_model_params(
                model, **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
    def add_random_joint_state(
        self,
        jt_key: jax.Array,
        qpos: jax.Array,
        minval: jax.Array,
        maxval: jax.Array,
    ):
        # TODO: This does not appropriately handle joints with no limits
        val = self._uniform(jt_key, qpos[self.qpos_free:].shape[0], minval=minval, maxval=maxval)
        val = self._np.clip(val + qpos[self.qpos_free:], *self._jt_lims)
        qpos = self._set_val_fn(qpos, val, min_idx=self.qpos_free, max_idx=None)
        return qpos

    def reset(
        self,
        rng: jax.Array,
        data: mjx.Data,
        history_length: int,
        num_resets: int = 0
    ) -> mjx_env.State:
        """Resets the environment to an initial state. Takes in a random key
        (rng) as input and outputs the first state of the environment."""

        # Initialize history buffers
        qpos_history       = self._np.zeros((history_length, self.nq))
        qvel_history       = self._np.zeros((history_length, self.nv))
        act_history        = self._np.zeros((history_length, self.action_size))

        # Set a simple info dict...
        info = {
            # Random variable
            'rng':                rng,
            
            # Basic proprioception
            'act_history':        act_history,
            'qpos_history':       qpos_history,
            'qvel_history':       qvel_history,
            'num_resets':         num_resets + 1
        }

        reward, done = self._np.zeros(2)
        obs, metrics = self._np.array([]), {}
        state = self._state_init_fn(data, obs, reward, done, metrics, info)

        return state
    
    @abstractmethod
    def step(self, state, action):
        """Steps the environment with an action outputted by the policy network.
        This function is run after policy evaluations and returns an Markov Decision
        Process state, which includes the observation for the next policy evaluation."""
        raise NotImplementedError()
    
    @abstractmethod
    def reward_function(
        data,
        action,
        info,
        done
    ):
        """Returns reward terms as a dictionary ({name: reward value}). These
        rewards are then summed in a weighted fashion based on their name and the
        weight values provided in your config."""
        raise NotImplementedError()
    
    @property
    def action_size(self):
        """Required action size for the environment"""
        raise NotImplementedError()
    
    @abstractmethod
    def _get_obs(
        self,
        data,
        info
    ):
        """Returns the observation given the data and info."""
        raise NotImplementedError()
    
    @property
    def observation_size(self):
        if self._np == jnp:
            return super().observation_size
        else:
            abstract_state = self.reset(jax.random.PRNGKey(0))
            obs = abstract_state.obs
            if type(obs) == dict:
                return len(obs['state'])
            return obs.shape
    
    def make_history(self, data, length):
        history = self._np.repeat(
            data[self._np.newaxis, :],
            length,
            axis=0
        )
        return history
    
    def update_history(self, history, new_data):
        history = self._np.roll(history, shift=1, axis=0)
        return self._set_val_fn(history, new_data, max_idx=1)
    
    def get_metrics(
        self,
        metrics,
        rewards
    ):
        weights = self.params.reward.weights
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = weights[k] * v

        return metrics

    def get_reward_and_metrics(
        self,
        rewards,
        metrics
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        _rewards = {k: v * self.params.reward.weights[k] for k, v in rewards.items()}
        reward = sum(_rewards.values())
        
        metrics = self.get_metrics(metrics, rewards)

        return reward, metrics
    
    # Generally useful rewards for RL....
    def reward_alive(self) -> jax.Array:
        """Reward for being alive."""
        return 1.0
    
    def reward_euclidean_imitation(
        self, qpos: jax.Array, reference: jax.Array, imitation_sigma: jax.Array
    ):
        """Reward for imitating. Can be applied to joints and flying frame."""
        deviation = self._np.linalg.norm(qpos - reference)
        error = self._np.square(deviation)
        return self.exp_reward(error, imitation_sigma)
    
    def reward_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ):
        """Reward for the rate of change of the action"""
        c1 = self._np.sqrt(self._np.sum(self._np.square(act - last_act)))
        c2 = self._np.sqrt(self._np.sum(self._np.square(act - 2 * last_act + last_last_act)))
        return c1 + c2
    
    def reward_vector_size(
        self, v: jax.Array, size_sigma: jax.Array
    ) -> jax.Array:
        mag2 = self._np.sum(self._np.square(v))
        return self.exp_reward(mag2, size_sigma)
    
    def exp_reward(self, mag2: jax.Array, sigma: jax.Array):
        return self._np.exp(-mag2 / sigma)
        
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model