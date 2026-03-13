import minimal_mjx as mm

def create_environment(config, for_training=False, **env_kwargs):
    env_params = mm.utils.config.create_config_dict(config['env_config'])
    backend = 'jnp' if for_training else config['backend']
    common_kwargs = {
        'backend': backend,
        'env_params': env_params
    }
    
    match config['env']:
        case 'Cheetah':
            from example.cheetah import Cheetah
            env = Cheetah(**common_kwargs, **env_kwargs)
        case _:
            raise Exception(f'Unknown enviornment {config["env"]}.')
    return env, env_params