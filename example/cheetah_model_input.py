"""Cheetah environment with the mjx.Model passed in as an explicit argument.

Unlike `example/cheetah.py` (which bakes a single `self._mjx_model` into its step
closure via `SwappableBase`, see `src/minimal_mjx/envs/generic/base.py:71`), this
lean env takes `model` as an argument to `init_data`/`step`. That makes the model a
vmappable axis, so many models (e.g. with different leg lengths) can be rolled out in
parallel with a single `jax.vmap`.

Only physics + base-frame extraction live here -- no reward/obs/State machinery.
"""

from pathlib import Path

import jax.numpy as jnp
import mujoco as mj
from mujoco import mjx
from mujoco_playground._src import mjx_env

XML = Path(__file__).resolve().parent / 'cheetah.xml'


class CheetahModelInput:
    """Cheetah physics with the mjx.Model passed in (so it can be vmapped)."""

    DEFAULT_FF = [0.0, 0.0, 0.0]   # rootx (x), rootz (z), rooty (pitch)
    DEFAULT_JT = [0.0] * 6         # the 6 actuated leg joints

    def __init__(self, n_substeps: int = 1):
        self._mj_model = mj.MjModel.from_xml_path(XML.as_posix())
        self.n_substeps = n_substeps
        self.torso_id = self._mj_model.body('torso').id   # == 1 (body 0 is world)
        self.nq = self._mj_model.nq
        self.nv = self._mj_model.nv
        self.nu = self._mj_model.nu

    def init_data(self, model: mjx.Model) -> mjx.Data:
        """Initialise mjx.Data at the default standing pose for `model`."""
        qpos = jnp.array(self.DEFAULT_FF + self.DEFAULT_JT)
        data = mjx_env.init(
            model,
            qpos=qpos,
            qvel=jnp.zeros(self.nv),
            ctrl=jnp.zeros(self.nu),
        )
        return data

    def step(self, model: mjx.Model, data: mjx.Data, action: jnp.ndarray) -> mjx.Data:
        """Advance the simulation one control step for `model`."""
        return mjx_env.step(model, data, action, self.n_substeps)

    def base_xpos(self, data: mjx.Data) -> jnp.ndarray:
        """Cartesian position (3,) of the torso (base) frame."""
        return data.xpos[self.torso_id]
