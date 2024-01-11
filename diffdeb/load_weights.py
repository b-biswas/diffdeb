import orbax

def load_model_weights(config):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(config.model_path, orbax_checkpointer)

    step = checkpoint_manager.latest_step() 
    params = checkpoint_manager.restore(step)["model"]["params"]

    return params