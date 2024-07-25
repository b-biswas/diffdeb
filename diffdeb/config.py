import galcheat
import jax.numpy as jnp
import ml_collections
import numpy as np
from galcheat.utilities import mean_sky_level

SURVEY_NAME = "LSST"

survey = galcheat.get_survey(SURVEY_NAME)
noise_sigma = []
for b, name in enumerate(survey.available_filters):
    filt = survey.get_filter(name)
    noise_sigma.append(np.sqrt(mean_sky_level(survey, filt).to_value("electron")))


def get_config_vae():
    """Get the default hyperparameter configuration for VAE."""
    config = ml_collections.ConfigDict()

    # User config
    config.model_path = "/pbs/throng/lsst/users/bbiswas/DiffDeblender/diffdeb/data/vae"

    # Normalization config
    config.linear_norm_coeff = 10000

    # Architecture config
    config.latent_dim = 16
    config.learning_rate = 1e-4
    config.kl_weight = 0.01
    config.input_shape = (
        45,
        45,
        len(survey.available_filters),
    )  # stamp size should be an odd number
    config.encoder_filters = (32, 128, 256, 512)
    config.decoder_filters = (64, 96, 128)
    config.encoder_kernels = (5, 5, 5, 5)
    config.decoder_kernels = (5, 5, 5)
    config.dense_layer_units = 512
    config.noise_sigma = jnp.asarray(noise_sigma) / config.linear_norm_coeff

    # training config
    config.num_epochs = 200
    config.steps_per_epoch_train = 1500
    config.steps_per_epoch_val = 500
    config.batch_size = 100
    config.patience = 10

    return config


def get_config_diffusion():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # User config
    config.model_path = "/pbs/throng/lsst/users/bbiswas/DiffDeblender/diffdeb/data/UNet"

    # Architecture config [NOT YET IMPLEMENTED]

    # TODO: Make architecture flexible.

    # training config
    config.linear_norm_coeff = 10000
    config.timesteps = 500
    config.num_epochs = 200
    config.steps_per_epoch_train = 1500
    config.steps_per_epoch_val = 500
    config.batch_size = 100
    config.learning_rate = 1e-4

    return config


def get_config_LDM():
    """Get the default hyperparameter configuration for LDM."""
    config = ml_collections.ConfigDict()
    config.vae_config = get_config_vae()
    config.diffusion_config = get_config_diffusion()
    config.exp_constant = 25
    config.t_min_val = 1e-3
    config.latent_scaling_factor = 0.5
    config.t_max_val = 0.5
    config.patience = 20

    if config.diffusion_config.linear_norm_coeff != config.vae_config.linear_norm_coeff:
        raise ValueError("Linear norm should be the same for both Encoder and UNet")

    return config
