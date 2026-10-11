# The config file

Every run is described by one YAML file. The example scripts read it through
[`Config`][minimal_mjx.utils.config.Config], which turns its sections into the plain
arguments the library functions take. Here is `example/cheetah.yaml`:

```yaml
--8<-- "example/cheetah.yaml"
```

## Top-level keys

| Key | Read as | Meaning |
| --- | --- | --- |
| `name` | `cfg.name` | Run name. The run directory is `save_dir/name`. A run named `test` may be overwritten and does not record a git hash. |
| `save_dir` | `cfg.save_dir` | Parent directory for all runs made from this config |
| `description` | | Free text for your own notes |
| `algorithm` | `cfg.algorithm` | Training algorithm; defaults to `ppo` |
| `env` | `cfg.env_name` | Environment class name, which `create_environment` maps to a class |
| `backend` | `cfg.backend` | `np` or `jnp` for rollouts. Training always uses `jnp`. |
| `git_hash` | | Not in your config: [`Config.save`][minimal_mjx.utils.config.Config.save] adds it to the copy saved in the run directory |

`cfg.run_dir` combines `save_dir` and `name`, so `example/cheetah.yaml` trains into
`example/results/cheetah/test/`.

## `env_config`

Everything under `env_config` is handed to the environment as `env_params`
([`cfg.env_params`][minimal_mjx.utils.config.Config.env_params]), and the environment
reads it as `self.params`. The library itself reads only these keys:

| Key | Meaning |
| --- | --- |
| `ctrl_dt` | Control period in seconds: how often the policy acts. This is `env.dt`. |
| `sim_dt` | Physics timestep. Each `step` runs `ctrl_dt / sim_dt` physics substeps. |
| `reward.weights` | One weight per reward term. The keys must match the names your `reward_function` returns. |

Any other key is yours to define and read in your environment. `Cheetah` reads
`action_scale` in `step`; the remaining keys in the example (`history_length`,
`ideal_base_height`, `termination`, ...) show the kind of settings an environment might
take, but `Cheetah` does not use them.

!!! warning "The config is locked"
    MuJoCo Playground locks `env_params` when the environment is built. Reading a key
    that is missing from the YAML raises an error, so add every key your environment
    reads to the config.

## `learning_params`

| Section | Read as | Meaning |
| --- | --- | --- |
| `ppo_params` | [`cfg.ppo_params`][minimal_mjx.utils.config.Config.ppo_params] | Passed as keyword arguments to Brax's PPO `train` function. See the [Brax PPO docs](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py) for every option. |
| `network_params` | [`cfg.network_params`][minimal_mjx.utils.config.Config.network_params] | Passed to Brax's `make_ppo_networks`, for example the hidden layer sizes |

A few PPO settings worth knowing:

- `num_timesteps`: total environment steps for the run. When resuming, this is the
  total budget, not the steps still to go.
- `num_envs`: environments simulated in parallel.
- `num_evals`: how many times training pauses to evaluate the policy. Each evaluation
  also writes a checkpoint and a row of `progress.csv`.
- `episode_length`: steps per episode before the environment is reset.
- `normalize_observations`: keep a running mean and variance of observations. The
  normalizer is saved inside each checkpoint.

## Using your own layout

Library functions never read the YAML directly. To use a different layout, subclass
`Config` and override the properties that differ:

```python
class MyConfig(mm.Config):
    @property
    def ppo_params(self) -> dict:
        return self.cfg.training.ppo.to_dict()
```

Or skip `Config` and pass `run_dir`, `ppo_params` and `network_params` to
[`train`][minimal_mjx.learning.training.train] yourself.
