# Examples

`example/` holds a complete task built on DeepMind Control's Cheetah. Its two scripts
cover the whole workflow:

| Script | What it does | Walkthrough |
| --- | --- | --- |
| `example/train.py` | Trains a PPO policy, checkpoints it, and saves a video of the result | [Training a policy](training.md) |
| `example/rollout.py` | Loads a checkpoint (or a dummy zero/random policy), rolls it out, and saves a video and reward plot | [Rolling out a policy](rollout.md) |

They share these supporting files:

| File | Purpose |
| --- | --- |
| `cheetah.xml` | The MuJoCo model |
| `cheetah.py` | `Cheetah(SwappableBase)`: `reset`, `step`, observations, rewards, termination. See [Writing an environment](../guides/environment.md). |
| `cheetah.yaml` | The run config: name, save directory, backend, `env_config`, `learning_params`. See [The config file](../guides/configuration.md). |
| `create_environment.py` | Maps the config's `env` name to an environment class |
| `cheetah_model_input.py` | A lean Cheetah that takes the `mjx.Model` as an argument, so many model variants can be vmapped |

Run every script from the repository root as a module (`python -m example.train ...`),
so that `from example.create_environment import ...` resolves.
