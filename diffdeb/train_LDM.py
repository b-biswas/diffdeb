import logging
import os
import shutil
import time
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax
from flax.training import orbax_utils, train_state
from jax import random

from diffdeb.diffusion import forward_noising
from diffdeb.load_weights import load_model_weights
from diffdeb.models import Encoder, UNet, reparameterize
from diffdeb.train_UNet import eval_f_UNet, train_step_UNet

logging.basicConfig(level=logging.INFO)


@partial(
    jax.jit,
    static_argnames=[
        "latent_dim",
        "encoder_filters",
        "encoder_kernels",
        "dense_layer_units",
    ],
)
def get_latent_images(
    params,
    batch,
    z_rng,
    latent_dim,
    encoder_filters,
    encoder_kernels,
    dense_layer_units,
):

    encoder_model = Encoder(
        latent_dim=latent_dim,
        filters=encoder_filters,
        kernels=encoder_kernels,
        dense_layer_units=dense_layer_units,
    )
    mean, logvar = encoder_model.apply({"params": params}, batch)

    latent_images = reparameterize(
        mean=mean,
        logvar=logvar,
        rng=z_rng,
    )

    return latent_images


def train_and_evaluate_LDM(
    train_tfds,
    val_tfds,
    config: ml_collections.ConfigDict,
):
    """Train and evaulate pipeline."""

    # STEP 1: Train VAE / Load trained weights.
    vae_params = load_model_weights(config.vae_config)

    # STEP 2: Train DM.
    @jax.jit
    def compute_av(cumulative, current, num_of_steps):
        return cumulative + current / num_of_steps

    rng = random.key(0)

    # Define checkpoint to save the trained model.
    if os.path.exists(config.diffusion_config.model_path):
        shutil.rmtree(config.diffusion_config.model_path)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        config.diffusion_config.model_path, orbax_checkpointer, options
    )

    # Initialize the UNet in Encoder latent space.
    logging.info("Initializing model.")

    rng, key = random.split(rng)
    latent_images = get_latent_images(
        params=vae_params["encoder"],
        batch=jnp.ones((1, 45, 45, 6), jnp.float32),
        z_rng=key,
        latent_dim=config.vae_config.latent_dim,
        encoder_filters=config.vae_config.encoder_filters,
        encoder_kernels=config.vae_config.encoder_kernels,
        dense_layer_units=config.vae_config.dense_layer_units,
    )
    init_data = (
        latent_images,
        jnp.ones((1), jnp.float32),
    )
    rng, key = random.split(rng)
    params_UNet = UNet().init(key, init_data)["params"]

    # Create train state.
    state = train_state.TrainState.create(
        apply_fn=UNet().apply,
        params=params_UNet,
        tx=optax.chain(
            optax.clip(0.1),
            optax.adam(
                learning_rate=optax.exponential_decay(
                    config.diffusion_config.learning_rate,
                    transition_steps=config.diffusion_config.steps_per_epoch_train * 30,
                    decay_rate=0.1,
                    end_value=1e-7,
                ),
            ),
        ),
    )

    # Training epochs.
    logging.info("start training...")
    min_val_loss = np.inf
    metrics = {"train loss": [], "val loss": []}

    for epoch in range(config.diffusion_config.num_epochs):
        start = time.time()
        train_ds = train_tfds.as_numpy_iterator()

        # Loop over training steps.
        current_epoch_train_loss = 0
        for _ in range(config.diffusion_config.steps_per_epoch_train):
            batch = next(train_ds)
            rng, key = random.split(rng)
            timestamps = random.randint(
                key,
                shape=(batch[0].shape[0],),
                minval=0,
                maxval=config.diffusion_config.timesteps,
            )

            # Generating the noise and noisy image for this batch.
            rng, key = random.split(rng)
            latent_batch = get_latent_images(
                params=vae_params["encoder"],
                batch=batch[0],
                z_rng=key,
                latent_dim=config.vae_config.latent_dim,
                encoder_filters=config.vae_config.encoder_filters,
                encoder_kernels=config.vae_config.encoder_kernels,
                dense_layer_units=config.vae_config.dense_layer_units,
            )
            rng, key = random.split(rng)
            noisy_images, noise = forward_noising(key, latent_batch, timestamps)

            # Train step.
            state, batch_train_loss = train_step_UNet(
                state, (noisy_images, latent_batch), timestamps
            )

            # Compute average loss in this epoch.
            current_epoch_train_loss = compute_av(
                current_epoch_train_loss,
                batch_train_loss,
                config.diffusion_config.steps_per_epoch_train,
            )
        metrics["train loss"].append(current_epoch_train_loss)

        # Loop over validation steps.
        metrics["val loss"].append(0.0)
        for _ in range(config.diffusion_config.steps_per_epoch_val):
            val_ds = val_tfds.as_numpy_iterator()
            batch = next(val_ds)
            rng, key = random.split(rng)
            timestamps = random.randint(
                key,
                shape=(latent_batch.shape[0],),
                minval=0,
                maxval=config.diffusion_config.timesteps,
            )

            # Generating the noise and noisy image for this batch.
            rng, key = random.split(rng)
            latent_batch = get_latent_images(
                params=vae_params["encoder"],
                batch=batch[0],
                z_rng=key,
                latent_dim=config.vae_config.latent_dim,
                encoder_filters=config.vae_config.encoder_filters,
                encoder_kernels=config.vae_config.encoder_kernels,
                dense_layer_units=config.vae_config.dense_layer_units,
            )
            rng, key = random.split(rng)
            noisy_images, noise = forward_noising(key, latent_batch, timestamps)

            # Eval step.
            batch_metrics = eval_f_UNet(
                state.params,
                (noisy_images, latent_batch),
                timestamps=timestamps,
            )

            # Compute average val loss in this epoch.
            metrics["val loss"][epoch] = compute_av(
                metrics["val loss"][epoch],
                batch_metrics["loss"],
                config.diffusion_config.steps_per_epoch_val,
            )

        logging.info(
            "\n\neval epoch: {}, train loss: {:.4f}, val loss: {:.4f}".format(
                epoch + 1,
                metrics["train loss"][epoch],
                metrics["val loss"][epoch],
            )
        )
        end = time.time()
        logging.info(f"\nTotal time taken: {end - start} seconds")
        if metrics["val loss"][epoch] < min_val_loss:
            logging.info(
                "\nVal loss improved from: {:.7f} to {:.7f}".format(
                    min_val_loss, metrics["val loss"][epoch]
                )
            )

            min_val_loss = metrics["val loss"][epoch]
            ckpt = {
                "model": state,
                "config": config.diffusion_config,
                "metrics": metrics,
            }

            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args})
