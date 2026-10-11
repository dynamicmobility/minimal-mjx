# minimal_mjx

Starter code for MJX-based reinforcement learning from the Dynamic Mobility Lab.
It is built on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
and [Brax](https://github.com/google/brax) PPO, and keeps training, checkpointing,
and evaluating policies simple:

- **Swappable backend.** Write an environment once and run it on JAX/MJX (`jnp`)
  for fast parallel training, or on NumPy/C MuJoCo (`np`) for fast, debuggable rollouts.
- **Base environment class** (`SwappableBase`) with common reward terms and helpers.
- **Config-agnostic library.** Library functions take plain arguments (`run_dir`,
  `ppo_params`, `network_params`, ...). `mm.utils.Config` is the only code that
  knows the YAML layout, so you can use your own layout by subclassing it or skip it.

## Install

```bash
pip install minimal_mjx            # from PyPI
pip install -e ".[dev]"            # from source, with pytest
```

## Repository layout

```
src/minimal_mjx/
  envs/generic/base.py    SwappableBase: backend switching, reward helpers, history buffers
  learning/training.py    train(): runs Brax PPO into a run dir, checkpoints, resumes
  learning/inference.py   load_policy(run_dir), get_last_model(run_dir), get_step_reset()
  eval/rollout.py         rollout_policy(): roll out a policy, record data, render video
  utils/config.py         Config (the only YAML-aware code), read_config(), save_config()
  utils/plotting.py       reward/data plotters, save_video(), save_metrics()
  utils/logging.py        Weights & Biases helpers (init, upload/download checkpoints)
  utils/state.py          EnvState, the NumPy-backend counterpart of mjx_env.State
example/                  a complete Cheetah task (see below)
tests/                    pytest suite
```

## Example: Cheetah

`example/` shows the full workflow on DeepMind Control's Cheetah:

| File | Purpose |
| --- | --- |
| `cheetah.xml` | MuJoCo model |
| `cheetah.py` | `Cheetah(SwappableBase)`: `reset`, `step`, observations, rewards, termination |
| `cheetah.yaml` | Run config: name, save dir, backend, `env_config`, `learning_params` |
| `create_environment.py` | Maps the config's `env` name to an environment class |
| `simulate.py` | Rolls out a zero-action policy to check the environment |
| `train.py` | Trains with PPO (logs to W&B; set `USE_WANDB = False` to turn this off) |
| `rollout.py` | Loads the latest checkpoint and renders a rollout |
| `cheetah_model_input.py` | Lean Cheetah that takes the `mjx.Model` as an argument, so many model variants can be vmapped (used by `scripts/rollout_cheetah_models.py`) |

Run each script from the repository root and pass the config as its argument:

```bash
python -m example.simulate example/cheetah.yaml   # video + reward plot in example/videos, example/plots
python -m example.train   --config example/cheetah.yaml   # checkpoints in <save_dir>/<name>/
python -m example.rollout --config example/cheetah.yaml   # roll out the latest checkpoint
```

Use `backend: np` for simulation and rollouts. Training always builds the `jnp`
version. Before training, simulate once with `backend: jnp` to check that your
environment is JAX-compatible.

To roll out an older policy, pass the `config.yaml` saved in that run's directory.
Its `git_hash` records the commit the run was trained at. Runs named `test` can be
overwritten and skip the git hash; any other name refuses to overwrite an existing run.

## Writing your own environment

Subclass `mm.envs.SwappableBase` and implement `reset`, `step`, `reward_function`,
`_get_obs`, and `action_size`. To stay backend-agnostic, use the backend handles
instead of `np`/`jnp` directly: `self._np`, `self._step_fn`, `self._data_init_fn`,
`self._set_val_fn`, `self._uniform`, and so on. Rewards come back as a dict
`{name: value}`, and `get_reward_and_metrics` weights them by
`env_config.reward.weights`. Use `state.info` to carry values from one `step` to
the next.

## Using the API directly

```python
import minimal_mjx as mm

cfg = mm.utils.Config('example/cheetah.yaml')   # cfg.env_name, cfg.env_params, cfg.run_dir, ...
env = Cheetah(env_params=cfg.env_params, backend=cfg.backend)

policy = mm.learning.inference.load_policy(cfg.run_dir)
frames, traj, rewards, data, info = mm.eval.rollout_policy(policy, env, n_steps=500)
mm.utils.plotting.save_video(frames, env.dt, 'videos/rollout.mp4')
```

Training follows the same pattern (see `example/train.py`):

```python
run_dir = mm.learning.create_training_directory(cfg.run_dir, overwrite=cfg.is_test)
cfg.save(run_dir / 'config.yaml', make_dir=False)      # records the git hash
mm.learning.train(env, env, run_dir, cfg.ppo_params, cfg.network_params,
                  algorithm=cfg.algorithm, overwrite=cfg.is_test)
```

For a different YAML layout, subclass `Config` and override its properties (as
moplayground's `MOConfig` does), or pass the arguments yourself.

## Tests

```bash
pytest            # fast suite (~15 s, CPU)
pytest -m slow    # rendering, video, and a tiny end-to-end PPO run
```
