import matplotlib
matplotlib.use('Agg')
import minimal_mjx as mm
from example.create_environment import create_environment

# Read configuration file
train_config = mm.utils.read_config()

# Create environment
env, env_cfg = create_environment(train_config, for_training=True)

# Initialize Weights and Biases Run object (if you want to use it)
USE_WANDB = False
if not USE_WANDB:
    run = None
else:
    name = train_config['save_dir'] + '/' + train_config['name']
    run = mm.utils.logging.initialize_wandb(
        name    = name.replace('/', ''),
        entity  = 'entity-name',
        project = 'project-name',
        config  = train_config
    )

# Begin training (the run directory and config are created inside train)
mm.learning.train(
    config=train_config,
    env=env,
    eval_env=env,
    run=run
)