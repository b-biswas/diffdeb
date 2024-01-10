import os 
import shutil 
import time
import logging

from functools import partial
import ml_collections

import numpy as np
import jax 
import jax.numpy as jnp
from jax import random 

import flax.linen as nn
from flax.training import train_state
from flax.training import orbax_utils
import orbax
import optax

from diffdeb.losses import kl_divergence, mse_loss_fn, vae_train_loss
from diffdeb.models import create_vae_model

logging.basicConfig(level=logging.INFO)


@jax.jit
def compute_metrics_vae(recon_x, x, mean, logvar, kl_weight):
  mse_loss = mse_loss_fn(prediction=recon_x, truth=x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'mse': mse_loss, 'kld': kld_loss, 'loss': mse_loss + kl_weight*kld_loss}

@partial(jax.jit, static_argnames=['latent_dim', 'input_shape', 'encoder_filters', 'encoder_kernels', 'decoder_filters', 'decoder_kernels', 'dense_layer_units'])
def train_step_vae(state, batch, z_rng, kl_weight, latent_dim, input_shape, encoder_filters, encoder_kernels, decoder_filters, decoder_kernels, dense_layer_units):
  def loss_fn(params):
    recon_x, mean, logvar = create_vae_model(
      latent_dim, 
      input_shape, 
      encoder_filters, 
      encoder_kernels, 
      decoder_filters,
      decoder_kernels, 
      dense_layer_units,
    ).apply({'params': params}, batch[0], z_rng)

    loss, mse_loss, kld_loss=vae_train_loss(
      prediction=recon_x, 
      truth=batch[1],
      mean=mean, 
      logvar=logvar,
      kl_weight=kl_weight,
    )
    return loss, (mse_loss, kld_loss)

  (loss, (mse_loss, kld_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

  return state.apply_gradients(grads=grads), (loss, mse_loss, kld_loss)

@partial(jax.jit, static_argnames=['latent_dim', 'input_shape', 'encoder_filters', 'encoder_kernels', 'decoder_filters', 'decoder_kernels', 'dense_layer_units'])
def eval_f_vae(params, images, z_rng, kl_weight, latent_dim, input_shape, encoder_filters, encoder_kernels, decoder_filters, decoder_kernels, dense_layer_units):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images[0], z_rng)
    metrics = compute_metrics_vae(recon_images, images[1], mean, logvar, kl_weight)
    return metrics

  return nn.apply(
    eval_model, create_vae_model(
      latent_dim, 
      input_shape, 
      encoder_filters, 
      encoder_kernels, 
      decoder_filters,
      decoder_kernels, 
      dense_layer_units,
    )
  )({'params': params})

def train_and_evaluate_vae(
    train_tfds,
    val_tfds,
    config: ml_collections.ConfigDict,
  ):
  """Train and evaulate pipeline."""
  @jax.jit
  def compute_av(cumulative, current, num_of_steps):
    return cumulative + current / num_of_steps
  
  # Define checkpoint
  if os.path.exists(config.model_path):
    shutil.rmtree(config.model_path)

  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
    config.model_path, orbax_checkpointer, options)
    
  rng = random.key(0)
  rng, key = random.split(rng)

  logging.info('Initializing model.')
  init_data = jnp.ones((config.batch_size, 45, 45, 6), jnp.float32)
  params = create_vae_model(
    config.latent_dim, 
    config.input_shape, 
    config.encoder_filters, 
    config.encoder_kernels, 
    config.decoder_filters,
    config.decoder_kernels, 
    config.dense_layer_units
  ).init(key, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=create_vae_model(
        config.latent_dim, 
        config.input_shape, 
        config.encoder_filters, 
        config.encoder_kernels, 
        config.decoder_filters,
        config.decoder_kernels, 
        config.dense_layer_units
      ).apply,
      params=params,
      tx=optax.chain(
      optax.clip(.1),
      optax.adam(
        learning_rate=optax.exponential_decay(
          config.learning_rate,
          transition_steps=config.steps_per_epoch_train*30,
          decay_rate=.1,
          end_value=1e-7,
        ),
      ),
    ),
  )
  
  min_val_loss = np.inf
  logging.info('Training started...')
  metrics = {"val loss": [], "val mse": [], "val kld": [], "train loss": [], "train mse": [], "train kld": []}
  for epoch in range(config.num_epochs):
    start = time.time()
    train_ds = train_tfds.as_numpy_iterator()
    metrics["train loss"].append(0.0)
    metrics["train kld"].append(0.0)
    metrics["train mse"].append(0.0)

    # Training loops 
    for _ in range(config.steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
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
      )
      
      for i, loss_name in enumerate(["loss", "mse", "kld"]):
        metrics[f"train {loss_name}"][epoch] = compute_av(
            metrics[f"train {loss_name}"][epoch], 
            batch_losses[i], 
            config.steps_per_epoch_train,
        )

    # Validation loops
    val_ds = val_tfds.as_numpy_iterator()
    for loss_name in ["loss", "mse", "kld"]:
      metrics["val "+loss_name].append(0.0)

    for _ in range(config.steps_per_epoch_val):
      batch = next(val_ds)
      rng, key = random.split(rng)
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
      )
      for loss_name in ["loss", "mse", "kld"]:
        metrics["val " + loss_name][epoch] = compute_av(
          metrics["val " + loss_name][epoch], 
          batch_metrics[loss_name], 
          config.steps_per_epoch_val,
        )

    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)
    logging.info(
        'eval epoch: {}, train loss: {:.7f}, train mse: {:.7f}, train kld: {:.7f}, val loss: {:.7f}, val MSE: {:.7f}, val kld: {:.7f}'.format(
            epoch + 1,
            metrics["train loss"][epoch],
            metrics["train mse"][epoch],
            metrics["train kld"][epoch],
            metrics["val loss"][epoch],
            metrics["val mse"][epoch], 
            metrics["val kld"][epoch],
        )
    )
    end = time.time()
    logging.info("\nTotal time taken: {} seconds".format(end - start))
    if jnp.isnan(metrics["val loss"][epoch]):
      logging.info("\nnan loss, terminating training")
      break
    if metrics["val loss"][epoch] < min_val_loss:
      logging.info("\nVal loss improved from: {:.7f} to {:.7f}".format(min_val_loss, metrics["val loss"][epoch]))
      min_val_loss = metrics["val loss"][epoch]
      ckpt = {'model': state, 'config': config, 'metrics': metrics}

      save_args = orbax_utils.save_args_from_target(ckpt)
      checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
      logging.info("Model saved at " + config.model_path)