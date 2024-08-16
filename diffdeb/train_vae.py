import logging
import os
import shutil
import time
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax
from flax.training import orbax_utils, train_state
from jax import random

from diffdeb.losses import vae_train_loss
from diffdeb.models import create_vae_model

logging.basicConfig(level=logging.INFO)


@partial(
    jax.jit,
    static_argnames=[
        "latent_dim",
        "input_shape",
        "encoder_filters",
        "encoder_kernels",
        "decoder_filters",
        "decoder_kernels",
        "dense_layer_units",
    ],
)
def train_step_vae(
    state,
    batch,
    z_rng,
    kl_weight,
    latent_dim,
    input_shape,
    encoder_filters,
    encoder_kernels,
    decoder_filters,
    decoder_kernels,
    dense_layer_units,
    noise_sigma,
    linear_norm_coeff,
):
    def loss_fn(params):
        recon_x, mean, logvar = create_vae_model(
            latent_dim,
            input_shape,
            encoder_filters,
            encoder_kernels,
            decoder_filters,
            decoder_kernels,
            dense_layer_units,
        ).apply({"params": params}, batch[0], z_rng)

        loss, reconst_loss, kld_loss = vae_train_loss(
            prediction=recon_x,
            truth=batch[1],
            mean=mean,
            logvar=logvar,
            kl_weight=kl_weight,
            noise_sigma=noise_sigma,
            linear_norm_coeff=linear_norm_coeff,
        )
        return loss, (reconst_loss, kld_loss)

    (loss, (reconst_loss, kld_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )

    return state.apply_gradients(grads=grads), (loss, reconst_loss, kld_loss)


@partial(
    jax.jit,
    static_argnames=[
        "latent_dim",
        "input_shape",
        "encoder_filters",
        "encoder_kernels",
        "decoder_filters",
        "decoder_kernels",
        "dense_layer_units",
    ],
)
def eval_f_vae(
    params,
    images,
    z_rng,
    kl_weight,
    latent_dim,
    input_shape,
    encoder_filters,
    encoder_kernels,
    decoder_filters,
    decoder_kernels,
    dense_layer_units,
    noise_sigma,
    linear_norm_coeff,
):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images[0], z_rng)
        loss, reconst_loss, kld_loss = vae_train_loss(
            prediction=recon_images,
            truth=images[1],
            mean=mean,
            logvar=logvar,
            kl_weight=kl_weight,
            noise_sigma=noise_sigma,
            linear_norm_coeff=linear_norm_coeff,
        )
        metrics = {
            "reconst": reconst_loss,
            "kld": kld_loss,
            "loss": reconst_loss + kl_weight * kld_loss,
        }
        return metrics

    return nn.apply(
        eval_model,
        create_vae_model(
            latent_dim,
            input_shape,
            encoder_filters,
            encoder_kernels,
            decoder_filters,
            decoder_kernels,
            dense_layer_units,
        ),
    )({"params": params})


def train_and_evaluate_vae(
    train_tfds,
    val_tfds,
    config: ml_collections.ConfigDict,
):
    """Train and evaulate pipeline."""
    rng = random.key(0)

    @jax.jit
    def compute_av(cumulative, current, num_of_steps):
        return cumulative + current / num_of_steps

    # Define checkpoint to save the trained model.
    if os.path.exists(config.model_path):
        shutil.rmtree(config.model_path)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        config.model_path, orbax_checkpointer, options
    )

    # Initialize the UNet in Encoder latent space.
    logging.info("Initializing model.")
    init_data = jnp.ones((1, 45, 45, 6), jnp.float32)
    rng, key = random.split(rng)
    params = create_vae_model(
        latent_dim=config.latent_dim,
        input_shape=config.input_shape,
        encoder_filters=config.encoder_filters,
        encoder_kernels=config.encoder_kernels,
        decoder_filters=config.decoder_filters,
        decoder_kernels=config.decoder_kernels,
        dense_layer_units=config.dense_layer_units,
    ).init(key, init_data, rng)["params"]

    # Create train state.
    state = train_state.TrainState.create(
        apply_fn=create_vae_model(
            config.latent_dim,
            config.input_shape,
            config.encoder_filters,
            config.encoder_kernels,
            config.decoder_filters,
            config.decoder_kernels,
            config.dense_layer_units,
        ).apply,
        params=params,
        tx=optax.chain(
            optax.clip(0.1),
            optax.adam(
                learning_rate=optax.exponential_decay(
                    config.learning_rate,
                    transition_steps=config.steps_per_epoch_train * 30,
                    decay_rate=0.1,
                    end_value=1e-7,
                ),
            ),
        ),
    )

    # Training epochs.
    min_val_loss = np.inf
    logging.info("Training started...")
    metrics = {
        "val loss": [],
        "val reconst": [],
        "val kld": [],
        "train loss": [],
        "train reconst": [],
        "train kld": [],
    }

    last_save_epoch = 0
    for epoch in range(config.num_epochs):
        start = time.time()
        train_ds = train_tfds.as_numpy_iterator()
        metrics["train loss"].append(0.0)
        metrics["train kld"].append(0.0)
        metrics["train reconst"].append(0.0)

        # Loop over training steps.
        for _ in range(config.steps_per_epoch_train):
            batch = next(train_ds)
            rng, key = random.split(rng)

            # Train step.
            state, batch_losses = train_step_vae(
                state,
                batch,
                z_rng=key,
                kl_weight=config.kl_weight,
                latent_dim=config.latent_dim,
                input_shape=config.input_shape,
                encoder_filters=config.encoder_filters,
                encoder_kernels=config.encoder_kernels,
                decoder_filters=config.decoder_filters,
                decoder_kernels=config.decoder_kernels,
                dense_layer_units=config.dense_layer_units,
                noise_sigma=config.noise_sigma,
                linear_norm_coeff=config.linear_norm_coeff,
            )

            # Compute average loss in this epoch.
            for i, loss_name in enumerate(["loss", "reconst", "kld"]):
                metrics[f"train {loss_name}"][epoch] = compute_av(
                    metrics[f"train {loss_name}"][epoch],
                    batch_losses[i],
                    config.steps_per_epoch_train,
                )

        # Loop over validation steps.
        val_ds = val_tfds.as_numpy_iterator()
        for loss_name in ["loss", "reconst", "kld"]:
            metrics["val " + loss_name].append(0.0)

        for _ in range(config.steps_per_epoch_val):
            batch = next(val_ds)
            rng, key = random.split(rng)

            # Eval step.
            batch_metrics = eval_f_vae(
                params=state.params,
                images=batch,
                z_rng=key,
                kl_weight=config.kl_weight,
                latent_dim=config.latent_dim,
                input_shape=config.input_shape,
                encoder_filters=config.encoder_filters,
                encoder_kernels=config.encoder_kernels,
                decoder_filters=config.decoder_filters,
                decoder_kernels=config.decoder_kernels,
                dense_layer_units=config.dense_layer_units,
                noise_sigma=config.noise_sigma,
                linear_norm_coeff=config.linear_norm_coeff,
            )

            # Compute average val loss in this epoch.
            for loss_name in ["loss", "reconst", "kld"]:
                metrics["val " + loss_name][epoch] = compute_av(
                    metrics["val " + loss_name][epoch],
                    batch_metrics[loss_name],
                    config.steps_per_epoch_val,
                )

        logging.info(
            "eval epoch: {}, train loss: {:.7f}, train reconst: {:.7f}, train kld: {:.7f}, val loss: {:.7f}, val reconst: {:.7f}, val kld: {:.7f}".format(
                epoch + 1,
                metrics["train loss"][epoch],
                metrics["train reconst"][epoch],
                metrics["train kld"][epoch],
                metrics["val loss"][epoch],
                metrics["val reconst"][epoch],
                metrics["val kld"][epoch],
            )
        )
        end = time.time()
        logging.info(f"\nTotal time taken: {end - start} seconds")
        if jnp.isnan(metrics["val loss"][epoch]):
            logging.info("\nnan loss, terminating training")
            break

        if metrics["val loss"][epoch] < min_val_loss:
            logging.info(
                "\nVal loss improved from: {:.7f} to {:.7f}".format(
                    min_val_loss, metrics["val loss"][epoch]
                )
            )
            last_save_epoch = epoch
            min_val_loss = metrics["val loss"][epoch]
            ckpt = {"model": state, "config": config, "metrics": metrics}

            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args})
            logging.info("Model saved at " + config.model_path)

        if epoch - last_save_epoch == config.patience:
            logging.info("Sorry running out of patience... ")
            break
