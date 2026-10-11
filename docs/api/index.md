# API reference

Every public function and class in `minimal_mjx`, generated from the docstrings in
`src/`. Each page covers one subpackage, split by the module the code lives in. Use the
search bar at the top to jump to any name.

| Subpackage | Modules | What it covers |
| --- | --- | --- |
| [`minimal_mjx.envs`](envs.md) | `generic.base` | `SwappableBase`, the base class for environments |
| [`minimal_mjx.learning`](learning.md) | `training`, `inference` | Training with Brax PPO, run directories, resuming; loading checkpoints and policies |
| [`minimal_mjx.eval`](eval.md) | `rollout`, `policy` | Rolling out a policy and recording it; dummy and open-loop policies |
| [`minimal_mjx.utils`](utils.md) | `config`, `logging`, `plotting`, `state`, `setupGPU`, `delete_failed_runs` | Configs, Weights & Biases, plots and videos, the NumPy state, GPU setup |

## Top-level shortcuts

The most used names are also importable from the package itself, so these are the same
function:

```python
import minimal_mjx as mm

mm.train
mm.learning.train
mm.learning.training.train
```

| Shortcut | Defined in |
| --- | --- |
| `mm.Config` | [`utils.config`](utils.md#minimal_mjx.utils.config) |
| `mm.SwappableBase` | [`envs.generic.base`](envs.md) |
| `mm.EnvState` | [`utils.state`](utils.md#minimal_mjx.utils.state) |
| `mm.train`, `mm.setup_ppo`, `mm.create_training_directory`, `mm.Resume`, `mm.find_resume`, ... | [`learning.training`](learning.md#minimal_mjx.learning.training) |
| `mm.load_policy`, `mm.get_last_model`, `mm.get_params`, `mm.get_step_reset`, ... | [`learning.inference`](learning.md#minimal_mjx.learning.inference) |
| `mm.rollout_policy`, `mm.make_dummy_inference_fn` | [`eval.rollout`](eval.md#minimal_mjx.eval.rollout) |
| `mm.make_zero_policy`, `mm.make_open_loop_policy`, ... | [`eval.policy`](eval.md#minimal_mjx.eval.policy) |
| `mm.initialize_wandb`, `mm.save_model`, `mm.download_model`, ... | [`utils.logging`](utils.md#minimal_mjx.utils.logging) |
| `mm.save_video`, `mm.save_metrics`, `mm.plot_progress`, ... | [`utils.plotting`](utils.md#minimal_mjx.utils.plotting) |
