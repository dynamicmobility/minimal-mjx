import minimal_mjx as mm


def create_environment(cfg: mm.utils.Config, for_training=False, **env_kwargs):
    """Build the environment named by ``cfg.env_name``. Training always uses ``jnp``."""
    env_params = cfg.env_params
    backend = 'jnp' if for_training else cfg.backend
    common_kwargs = {
        'backend': backend,
        'sim_dt': env_params.sim_dt,
        'ctrl_dt': env_params.ctrl_dt,
        'reward_weights': env_params.reward.weights.to_dict(),
    }

    match cfg.env_name:
        case 'Cheetah':
            from example.cheetah import Cheetah
            env = Cheetah(
                **common_kwargs,
                action_scale=env_params.action_scale,
                **env_kwargs,
            )
        case _:
            raise Exception(f'Unknown environment {cfg.env_name}.')
    return env, env_params
