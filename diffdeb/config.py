
import ml_collections
import numpy as np
from typing import Sequence


def get_config_vae():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # architecture config
    config.latents = 20

    config.linear_norm_coeff = 10000

    # training config
    config.num_epochs = 30
    config.steps_per_epoch_train = 1500
    config.steps_per_epoch_val = 500
    config.batch_size=100
    config.latent_dim=16
    config.learning_rate=1e-4
    config.input_shape=(45, 45, 6) # stamp size should be an odd number
    config.encoder_filters=(32, 64, 128, 256)
    config.decoder_filters=(16, 32, 64)
    config.encoder_kernels=(5, 5, 5, 5)
    config.decoder_kernels=(5, 5, 5)
    config.dense_layer_units=128
    config.model_path = "/pbs/throng/lsst/users/bbiswas/DiffDeblender/diffdeb/data/vae"

    return config

def get_config_diffusion():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.linear_norm_coeff = 10000
  config.timesteps = 200
  config.num_epochs = 50
  config.steps_per_epoch_train = 1500
  config.steps_per_epoch_val = 500
  config.batch_size=100
  config.learning_rate = 1e-4
  config.model_path = "/pbs/throng/lsst/users/bbiswas/DiffDeblender/diffdeb/data/UNet"
  

  return config

