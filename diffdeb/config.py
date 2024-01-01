
import ml_collections


def get_config_vae():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # architecture config
    config.latents = 20

    config.linear_norm_coeff = 10000

    # training config
    config.num_epochs = 5
    config.steps_per_epoch_train = 10
    config.steps_per_epoch_val = 10
    config.batch_size=32
    config.latent_dim=16
    config.learning_rate=1e-4
    config.model_path = "/pbs/throng/lsst/users/bbiswas/DiffDeblender/diffdeb/data/vae"

    return config

def get_config_diffusion():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.linear_norm_coeff = 10000
  config.timesteps = 200
  config.batch_size = 3
  config.num_epochs = 10
  config.learning_rate = 1e-4
  config.steps_per_epoch_train = 10
  config.steps_per_epoch_val = 10
  config.model_path = "/pbs/throng/lsst/users/bbiswas/DiffDeblender/diffdeb/data/UNet"
  

  return config

