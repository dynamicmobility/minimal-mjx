# Writing an environment

An environment subclasses [`SwappableBase`][minimal_mjx.envs.generic.base.SwappableBase].
You write it once, and it runs on either backend:

- **`jnp`** uses JAX and MJX. Brax trains thousands of copies in parallel on the
  accelerator, so training always uses this backend.
- **`np`** uses NumPy and MuJoCo's C bindings. It is fast for a single environment, and
  ordinary Python debugging works: breakpoints, prints, `if` statements on values.

This page walks through `example/cheetah.py` to show what you need to implement.

## Backend handles

The trick that makes one class run on both backends: never call `np`, `jnp`, `mjx` or
`jax.random` directly. `SwappableBase` sets up handles that point at the right one:

| Handle | `jnp` backend | `np` backend |
| --- | --- | --- |
| `self._np` | `jax.numpy` | `numpy` |
| `self._step_fn(data, ctrl)` | `mjx_env.step`, with the configured substeps | `mj_step` on a copy of `data` |
| `self._data_init_fn(qpos, qvel, ctrl, time, xfrc_applied)` | `mjx.make_data` + `mjx.forward` | `MjData` + `mj_forward` |
| `self._state_init_fn(data, obs, reward, done, metrics, info)` | `mjx_env.State` | [`EnvState`][minimal_mjx.utils.state.EnvState] |
| `self._split(rng, n)` | `jax.random.split` | yields `None` |
| `self._uniform`, `self._normal` | `jax.random` | `np.random` |
| `self._set_val_fn(arr, val, min_idx, max_idx)` | `arr.at[...].set(val)` | copy, then assign |
| `self._cond(pred, true_fn, false_fn, x)` | `jax.lax.cond` | Python `if` |

## What to implement

### `__init__`

Pass the MuJoCo XML, the `env_params` from the config, and the backend:

```python
class Cheetah(mm.envs.SwappableBase):
    def __init__(self, env_params, backend):
        super().__init__(
            xml_path   = CheetahInterface.XML,
            env_params = env_params,
            backend    = backend,
        )
```

`SwappableBase` compiles the model, sets its timestep to `env_config.sim_dt`, and
builds the backend handles. `num_free` (default 3) is the number of `qpos` entries
taken by the floating base; the Cheetah's are `rootx`, `rootz` and `rooty`.

### `reset(rng)`

Build the initial `data`, compute the first observation and rewards, and return a state:

```python
def reset(self, rng):
    rng, qpos_key, qvel_key = self._split(rng, 3)
    qpos = self._np.hstack([CheetahInterface.DEFAULT_FF, CheetahInterface.DEFAULT_JT])
    ...
    data = self._data_init_fn(qpos=qpos, qvel=qvel, ctrl=ctrl, time=0.0,
                              xfrc_applied=self._np.zeros((self._mj_model.nbody, 6)))
    info = {'xposbefore': 0.0, 'xposafter': 0.01, 'ang': 0.0}
    rewards = self.reward_function(data=data, action=ctrl, info=info, done=False)
    reward, metrics = self.get_reward_and_metrics(rewards, {})
    obs = self._get_obs(data, info)
    return self._state_init_fn(data, obs, reward, done, metrics, info)
```

Under JAX, every key in `info` and `metrics` must already exist after `reset`, because
the state's structure cannot change between steps. Initialize anything `step` will
write here.

### `step(state, action)`

Apply the action, advance the physics, and return the next state:

```python
def step(self, state, action):
    state.info['xposbefore'] = state.data.qpos[0]
    action = self._np.clip(self.params.action_scale * action, -1.0, 1.0)
    data = self._step_fn(state.data, action)
    state.info['xposafter'] = data.qpos[0]
    state.info['ang']       = data.qpos[2]

    done = self.fall_termination(state.info)
    rewards = self.reward_function(data=data, action=action, info=state.info, done=done)
    reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
    obs = self._get_obs(data, state.info)
    return self._state_init_fn(data, obs, reward, done.astype(float), metrics, state.info)
```

Use `state.info` to carry values from one step to the next, as the Cheetah does with
its x position to compute forward velocity.

### `reward_function(data, action, info, done)`

Return a dict of named reward terms:

```python
def reward_function(self, data, action, info, done):
    return {
        'alive'  : self.reward_alive(),
        'energy' : self.reward_energy(action),
        'height' : self.reward_height(data),
        'run'    : self.reward_run(info),
        'done'   : self.reward_done(done),
    }
```

[`get_reward_and_metrics`][minimal_mjx.envs.generic.base.SwappableBase.get_reward_and_metrics]
multiplies each term by `env_config.reward.weights[name]` and sums them into the scalar
reward. It also logs each weighted term as `reward/<name>` in `metrics`, which is what
appears in W&B and in the rollout reward plot. Every name you return needs a weight in
the config; set a weight to `0` to log a term without rewarding it.

`SwappableBase` provides some common terms:
[`reward_alive`][minimal_mjx.envs.generic.base.SwappableBase.reward_alive],
[`reward_action_rate`][minimal_mjx.envs.generic.base.SwappableBase.reward_action_rate],
[`reward_euclidean_imitation`][minimal_mjx.envs.generic.base.SwappableBase.reward_euclidean_imitation],
[`reward_vector_size`][minimal_mjx.envs.generic.base.SwappableBase.reward_vector_size] and
[`exp_reward`][minimal_mjx.envs.generic.base.SwappableBase.exp_reward].

### `_get_obs(data, info)` and `action_size`

```python
def _get_obs(self, data, info):
    obs = self._np.concatenate([data.qpos[1:], self._np.clip(data.qvel, -10.0, 10.0)])
    return {'state': obs, 'privileged_state': obs}

@property
def action_size(self):
    return 6
```

Returning a dict lets the policy and the value function see different inputs. By
default both read `obs['state']`. To give the critic information the robot would not
have at deployment (asymmetric actor-critic), put it in `privileged_state` and set
`value_obs_key: privileged_state` under `network_params` in the config.

For observations that need recent history, `SwappableBase` has
[`make_history`][minimal_mjx.envs.generic.base.SwappableBase.make_history] and
[`update_history`][minimal_mjx.envs.generic.base.SwappableBase.update_history].

## Registering the environment

The example scripts find environments by name through `example/create_environment.py`.
Add a case for yours:

```python
match cfg.env_name:
    case 'Cheetah':
        from example.cheetah import Cheetah
        env = Cheetah(**common_kwargs, **env_kwargs)
    case 'MyRobot':
        from example.my_robot import MyRobot
        env = MyRobot(**common_kwargs, **env_kwargs)
```

Then set `env: MyRobot` in your config, and check it with a zero-action rollout on both
backends before training (see [Rolling out a policy](../examples/rollout.md)).
