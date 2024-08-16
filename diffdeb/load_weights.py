import orbax
from flax.training import orbax_utils


def load_model_weights(config):

    print(f"Loading model weights from {config.model_path}")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        config.model_path, orbax_checkpointer, options
    )

    step = checkpoint_manager.latest_step()

    print(f"Loading model weights from {config.model_path} and step {step}")
    params = checkpoint_manager.restore(step)["model"]["params"]

    return params
