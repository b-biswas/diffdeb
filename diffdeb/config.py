import ml_collections


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
    config.kl_weight = 0.001
    config.input_shape = (45, 45, 6)  # stamp size should be an odd number
    config.encoder_filters = (32, 64, 128)
    config.decoder_filters = (16, 32, 64)
    config.encoder_kernels = (5, 5, 5)
    config.decoder_kernels = (5, 5, 5)
    config.dense_layer_units = 0

    # training config
    config.num_epochs = 30
    config.steps_per_epoch_train = 1500
    config.steps_per_epoch_val = 500
    config.batch_size = 100

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
    config.timesteps = 200
    config.num_epochs = 50
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

    if config.diffusion_config.linear_norm_coeff == config.vae_config.linear_norm_coeff:
        raise ValueError("Linear norm should be the same for both Encoder and UNet")

    return config
