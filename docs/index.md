# minimal_mjx

Starter code for MJX-based reinforcement learning from the Dynamic Mobility Lab.
It is built on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
and [Brax](https://github.com/google/brax) PPO. It keeps training, checkpointing and
evaluating policies simple.

<figure markdown="span">
  <video src="assets/cheetah-rollout.mp4" poster="assets/cheetah-rollout-poster.jpg"
         width="360" autoplay loop muted playsinline controls></video>
  <figcaption>The example Cheetah policy after 50M environment steps (about 8 minutes
  of training).</figcaption>
</figure>

## What you get

- **A swappable backend.** Write an environment once. Run it on JAX/MJX (`jnp`) for
  fast parallel training, or on NumPy and the C MuJoCo bindings (`np`) for fast,
  debuggable rollouts.
- **A base environment class**, [`SwappableBase`][minimal_mjx.envs.generic.base.SwappableBase],
  with common reward terms and helpers.
- **A config-agnostic library.** Library functions take plain arguments (`run_dir`,
  `ppo_params`, `network_params`, ...). [`Config`][minimal_mjx.utils.config.Config] is
  the only code that knows the YAML layout, so you can use your own layout by
  subclassing it, or skip it entirely.

## Install

```bash
pip install minimal_mjx            # from PyPI
pip install -e ".[dev]"            # from source, with pytest
```

## Quick start

From the repository root:

```bash
# Train the example Cheetah into example/results/cheetah/test/
python -m example.train   --config example/cheetah.yaml --no_wandb

# Roll out the latest checkpoint and save a video and reward plot
python -m example.rollout --config example/cheetah.yaml
```

## Where to go next

<div class="grid cards" markdown>

-   **[Training a policy](examples/training.md)**

    A step-by-step walk through `example/train.py`, and what a finished run directory
    looks like.

-   **[Rolling out a policy](examples/rollout.md)**

    Load a checkpoint, roll it out on the NumPy backend, and save a video and reward
    plot.

-   **[Writing an environment](guides/environment.md)**

    How `Cheetah` subclasses `SwappableBase`, and what you need to implement for your
    own robot.

-   **[API reference](api/index.md)**

    Every public function and class, organized by module. Use the search bar to
    jump to any name.

</div>
