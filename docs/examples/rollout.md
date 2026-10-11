# Rolling out a policy

`example/rollout.py` loads a trained policy, runs it in the environment, and saves a
video and a plot of every reward term. It can also drive the environment with zero or
random actions, which is the quickest way to check a new environment before training
anything.

```bash
python -m example.rollout --config example/cheetah.yaml                       # latest checkpoint
python -m example.rollout --config example/cheetah.yaml --inference_fn zero   # no policy, zero actions
python -m example.rollout --config example/results/cheetah/test/config.yaml \
    --model_path example/results/cheetah/test/000025016320                     # a specific checkpoint
```

## Command-line options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(required)* | Path to the run's YAML config |
| `--inference_fn` | `policy` | `policy` loads a trained checkpoint; `zero` or `random` send dummy actions instead |
| `--model_path` | latest | A checkpoint directory to load instead of the newest one in the run directory |
| `--T` | `10.0` | Rollout length in seconds |
| `--width`, `--height` | `640`, `480` | Video size in pixels |
| `--camera` | `track` | Camera to render from |
| `--stochastic` | off | Sample actions from the policy instead of taking the mean action |
| `--seed` | none | Seed NumPy's random generator, which picks the initial state's random key, so rollouts repeat |

## Step by step

### 1. Set up rendering and keep JAX on the CPU

```python
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS'] = 'cpu'
```

`egl` renders without a display. `JAX_PLATFORMS=cpu` keeps JAX off the GPU: the
rollout steps a single environment, usually on the NumPy backend, so the only JAX work
is evaluating the small policy network, which is fast on the CPU.

### 2. Read the config and build the environment

```python
cfg = mm.Config(config_path)
env, env_params = create_environment(cfg)
```

Unlike training, this builds the environment on the backend the config names. The
example config uses `backend: np`, which steps the C MuJoCo bindings directly. That is
fast for a single environment, and you can set breakpoints and print values inside
`reset` and `step`, which JAX's tracing does not allow.

!!! tip "Check your environment is JAX-compatible"
    Training always uses `jnp`. Before training a new environment, run this script once
    with `backend: jnp` in the config and `--inference_fn zero`. Code that works on
    NumPy but not under JAX (Python `if` on array values, in-place array writes) fails
    here quickly, instead of after you have queued a training job.

To roll out an older run, pass the `config.yaml` saved in that run's directory. Its
`save_dir` and `name` point back at the same run, and its `git_hash` records the
commit the policy was trained at.

### 3. Build the inference function

```python
if inference_fn_type == 'policy':
    inference_fn = mm.learning.inference.load_policy(
        cfg.run_dir,
        deterministic   = deterministic,
        checkpoint_path = model_path
    )
    inference_fn = jax.jit(inference_fn)
else:
    inference_fn = mm.make_dummy_inference_fn(env=env, mode=inference_fn_type)
```

An inference function maps `(obs, rng)` to `(action, extras)`, the same signature
Brax uses.

- [`load_policy`][minimal_mjx.learning.inference.load_policy] restores a checkpoint: the
  one at `checkpoint_path` if given, otherwise the newest step directory in `run_dir`
  (found with [`get_last_model`][minimal_mjx.learning.inference.get_last_model]). Each
  checkpoint carries its network config, so nothing about the architecture has to
  match the current YAML. `deterministic=True` takes the mean action. The result is
  jitted so each call is a single compiled function.
- [`make_dummy_inference_fn`][minimal_mjx.eval.rollout.make_dummy_inference_fn] returns
  a policy that sends all zeros (`zero`) or uniform random actions in [-1, 1]
  (`random`).

### 4. Roll out

```python
frames, _, reward_plotter, _, _ = mm.eval.rollout_policy(
    inference_fn = inference_fn,
    env          = env,
    n_steps      = int(T / env.dt),
    camera       = camera,
    width        = width,
    height       = height
)
```

[`rollout_policy`][minimal_mjx.eval.rollout.rollout_policy]:

1. Gets `step` and `reset` functions from
   [`get_step_reset`][minimal_mjx.learning.inference.get_step_reset], jitted on the `jnp`
   backend and plain Python on `np`.
2. Resets the environment with a random key drawn from NumPy (which is why `--seed`
   makes rollouts repeat).
3. Steps `n_steps` times, or until the environment reports `done`. Each step it records
   MuJoCo data (`qpos`, `qvel`, `ctrl`, ...) in a
   [`MujocoPlotter`][minimal_mjx.utils.plotting.MujocoPlotter], reward terms in a
   [`RewardPlotter`][minimal_mjx.utils.plotting.RewardPlotter], and chosen `info`
   fields in an [`InfoPlotter`][minimal_mjx.utils.plotting.InfoPlotter].
4. Renders every state to a frame.

It returns `(frames, trajectory, reward_plotter, data_plotter, info_plotter)`. This
script keeps the frames and the reward plotter. The other three are there for your
own analysis: the plotters can write their data to HDF5 with `save_to_h5`.

### 5. Save the video and reward plot

```python
mm.utils.plotting.save_video(frames, env.dt, Path(f'example/videos/{cfg.env_name}-rollout.mp4'))
mm.utils.plotting.save_metrics(reward_plotter, Path(f'example/plots/{cfg.env_name}-reward.pdf'))
```

[`save_video`][minimal_mjx.utils.plotting.save_video] writes the frames at the control
rate (`1 / env.dt` frames per second), so the video plays in real time.
[`save_metrics`][minimal_mjx.utils.plotting.save_metrics] plots each weighted reward
term and the total reward over the rollout, one subplot each.

!!! note
    If `example/videos/` or `example/plots/` does not exist, both functions ask on the
    terminal whether to create it, and skip saving if you answer no.

## Full script

??? example "example/rollout.py"

    ```python
    --8<-- "example/rollout.py"
    ```
