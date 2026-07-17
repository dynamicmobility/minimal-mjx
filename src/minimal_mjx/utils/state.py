from dataclasses import dataclass
import mujoco
import numpy as np
import jax

@dataclass
class EnvState:
    data: mujoco.MjData
    obs: np.ndarray
    reward: float
    done: bool
    metrics: dict
    info: dict
    
    def replace(self, **kwargs):
        """Replace fields in the MujocoState."""
        return EnvState(
            data=kwargs.get('data', self.data),
            obs=kwargs.get('obs', self.obs),
            reward=kwargs.get('reward', self.reward),
            done=kwargs.get('done', self.done),
            metrics=kwargs.get('metrics', self.metrics),
            info=kwargs.get('info', self.info)
        )
        
def get_batch_shape(obs) -> tuple:
    """Gets the batch size of an observation Pytree. 
    
    Returns () if obs came from a single env, and (num_envs,) if obs came from a 
    batched rollout.
    """
    return jax.tree_util.tree_leaves(obs)[0].shape[:-1]

