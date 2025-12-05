from pathlib import Path
import mujoco as mj
from mujoco_playground._src import mjx_env
from mujoco import mjx
import jax

class CheetahInterface:
    XML         = Path('envs/dmcontrol/model/cheetah.xml')
    DEFAULT_FF  = [0.0, 0.0, 0.0]
    DEFAULT_JT  = [0.0] * 6