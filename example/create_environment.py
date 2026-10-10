import minimal_mjx as mm


def create_environment(cfg: mm.utils.Config, for_training=False, **env_kwargs):
    """Build the environment named by ``cfg.env_name``. Training always uses ``jnp``."""
    env_params = cfg.env_params
    backend = 'jnp' if for_training else cfg.backend
    common_kwargs = {
        'backend': backend,
        'env_params': env_params
    }

    match cfg.env_name:
        case 'Cheetah':
            from example.cheetah import Cheetah
            env = Cheetah(**common_kwargs, **env_kwargs)
        case _:
            raise Exception(f'Unknown environment {cfg.env_name}.')
    return env, env_params
