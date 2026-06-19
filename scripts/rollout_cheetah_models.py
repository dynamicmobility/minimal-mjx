"""Roll out N cheetahs with different back-thigh ('bthigh') lengths in parallel.

Each model differs only in the length of the first back-leg link, so they share
topology and stack into one batched `mjx.Model`. A single `jax.vmap` rolls the whole
batch out under identical sinusoidal actions; we plot each base-frame (torso) x/z vs time.

Runs on GPU by default. Force CPU with: JAX_PLATFORMS=cpu python scripts/rollout_cheetah_models.py
"""

import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# make `example` importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import mjx

from example.cheetah_model_input import CheetahModelInput, XML

N = 1024        # number of cheetah variants
T = 50          # rollout length (control steps)
SIGMA = 0.15    # std of the leg-length scale (Normal(1.0, SIGMA))
SEED = 0
AMP = 0.8       # action amplitude (ctrl range is [-1, 1])
FREQ = 1.5      # action frequency [Hz]
OUT_DIR = Path('scripts/outputs')


def make_model(scale: float) -> mjx.Model:
    """Recompile the cheetah with the back-thigh link scaled, then upload to MJX.
    A length change needs a recompile (kinematic tree + inertia are derived), so we
    edit an MjSpec rather than mjx.Model leaves. Total mass stays 14 (settotalmass)."""
    spec = mj.MjSpec.from_file(XML.as_posix())
    g = spec.geom('bthigh')
    g.size = [g.size[0], g.size[1] * scale, g.size[2]]   # lengthen the thigh capsule
    b = spec.body('bshin')
    b.pos = [c * scale for c in b.pos]                   # move child attachment to match
    return mjx.put_model(spec.compile())


def stack_models(models: list[mjx.Model]) -> mjx.Model:
    """Stack same-topology models along a new leading batch axis. Independently compiled
    models don't share a treedef (a few static fields, e.g. the unused height-field bound,
    track geometry), so we stack only the traced leaves and reuse the first model's treedef."""
    leaves = [jax.tree_util.tree_leaves(m) for m in models]
    _, treedef = jax.tree_util.tree_flatten(models[0])
    return jax.tree_util.tree_unflatten(treedef, [jnp.stack(col) for col in zip(*leaves)])


def build_batched_model(scales: np.ndarray, workers: int) -> mjx.Model:
    """Build + stack all N models. compile()/put_model() release the GIL, so threads help."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        models = list(ex.map(make_model, [float(s) for s in scales]))
    return stack_models(models)


def main(workers: int = 8) -> None:
    env = CheetahModelInput(n_substeps=1)
    env_dt = 0.01   # ctrl_dt from example/cheetah.yaml (n_substeps=1)

    # 1. sample leg-length scales and build/stack the batched model
    rng = np.random.default_rng(SEED)
    scales = np.clip(rng.normal(1.0, SIGMA, size=N), a_min=0.5, a_max=None)
    batched_model = build_batched_model(scales, workers)

    # 2. identical sinusoidal actions for every model (function of time only)
    phases = jnp.linspace(0.0, jnp.pi, env.nu)
    sin_action = lambda t: AMP * jnp.sin(2.0 * jnp.pi * FREQ * t + phases)

    # 3. vmapped rollout, recording torso [x, z] each step
    def rollout(model):
        def body(d, i):
            d = env.step(model, d, sin_action(i * env_dt))
            return d, env.base_xpos(d)[jnp.array([0, 2])]   # x, z
        _, xz = jax.lax.scan(body, env.init_data(model), jnp.arange(T))
        return xz   # (T, 2)

    traj = np.asarray(jax.jit(jax.vmap(rollout))(batched_model))   # (N, T, 2)
    time = np.arange(T) * env_dt

    # 4. save trajectories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / 'cheetah_leglength_traj.npz', time=time, xpos=traj, scales=scales)

    # 5. plot x and z vs time (one translucent line per model)
    fig, (ax_x, ax_z) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_x.plot(time, traj[:, :, 0].T, color='C0', alpha=0.1)
    ax_z.plot(time, traj[:, :, 1].T, color='C0', alpha=0.1)
    ax_x.set_ylabel('base x [m]')
    ax_z.set_ylabel('base z [m]')
    ax_z.set_xlabel('time [s]')
    ax_x.set_title(f'{N} cheetahs, back-thigh length varied ~ Normal(1.0, {SIGMA})')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'cheetah_leglength_xz.png', dpi=150)
    plt.close(fig)

    print(f'traj shape {traj.shape} -> {OUT_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8,
                        help='thread-pool size for building models')
    args = parser.parse_args()
    main(workers=args.workers)
