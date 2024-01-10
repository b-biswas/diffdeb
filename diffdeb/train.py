from diffdeb.losses import kl_divergence, mse_loss_fn, vae_train_loss
from diffdeb.models import create_vae_model, UNet
import jax 
import jax.numpy as jnp
import numpy as np
from jax import random 
import logging
from diffdeb.diff_utils import forward_noising
import time
from functools import partial

import ml_collections

import flax.linen as nn
from flax.training import train_state
import orbax
from flax.training import orbax_utils
import optax

logging.basicConfig(level=logging.INFO)

@jax.jit
def compute_metrics_vae(recon_x, x, mean, logvar):
  mse_loss = mse_loss_fn(prediction=recon_x, truth=x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'mse': mse_loss, 'kld': kld_loss, 'loss': mse_loss + kld_loss}

@partial(jax.jit, static_argnames=['latent_dim', 'input_shape', 'encoder_filters', 'encoder_kernels', 'decoder_filters', 'decoder_kernels', 'dense_layer_units'])
def train_step_vae(state, batch, z_rng, latent_dim, input_shape, encoder_filters, encoder_kernels, decoder_filters, decoder_kernels, dense_layer_units):
  def loss_fn(params):
    recon_x, mean, logvar = create_vae_model(
      latent_dim, 
      input_shape, 
      encoder_filters, 
      encoder_kernels, 
      decoder_filters,
      decoder_kernels, 
      dense_layer_units
    ).apply({'params': params}, batch[0], z_rng)

    loss=vae_train_loss(
      prediction=recon_x, 
      truth=batch[1],
      mean=mean, 
      logvar=logvar,
    )
    return loss

  loss, grads = jax.value_and_grad(loss_fn, argnums=0)(state.params)
  return state.apply_gradients(grads=grads), loss.mean()

@partial(jax.jit, static_argnames=['latent_dim', 'input_shape', 'encoder_filters', 'encoder_kernels', 'decoder_filters', 'decoder_kernels', 'dense_layer_units'])
def eval_f_vae(params, images, z_rng, latent_dim, input_shape, encoder_filters, encoder_kernels, decoder_filters, decoder_kernels, dense_layer_units):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images[0], z_rng)
    # comparison = jnp.concatenate([
    #     images[1].reshape(-1, 45, 45, 6),
    #     recon_images.reshape(-1, 45, 45, 6),
    # ])

    # generate_images = vae.generate(z)
    # generate_images = generate_images.reshape(-1, 45, 45, 6)
    metrics = compute_metrics_vae(recon_images, images[1], mean, logvar)
    #return metrics, comparison, generate_images
    return metrics

  return nn.apply(eval_model, create_vae_model(
      latent_dim, 
      input_shape, 
      encoder_filters, 
      encoder_kernels, 
      decoder_filters,
      decoder_kernels, 
      dense_layer_units,
    ))({'params': params})

def train_and_evaluate_vae(
    train_tfds,
    val_tfds,
    config: ml_collections.ConfigDict,
  ):
  """Train and evaulate pipeline."""
  @jax.jit
  def compute_av(cumulative, current, num_of_steps):
    return cumulative + current / num_of_steps
  
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
  metrics = {"val loss": [], "val mse": [], "val kld": [], "train loss": []}
  for epoch in range(config.num_epochs):
    train_ds = train_tfds.as_numpy_iterator()
    start = time.time()
    metrics["train loss"].append(0.0)
    for _ in range(config.steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state, batch_train_loss = train_step_vae(
        state, 
        batch, 
        key, 
        config.latent_dim, 
        input_shape=config.input_shape,
        encoder_filters=config.encoder_filters,
        encoder_kernels=config.encoder_kernels,
        decoder_filters=config.decoder_filters,
        decoder_kernels=config.decoder_kernels,
        dense_layer_units=config.dense_layer_units,
      )
      metrics["train loss"][epoch] = compute_av(
        metrics["train loss"][epoch], 
        batch_train_loss, 
        config.steps_per_epoch_train,
      )
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
        latent_dim=config.latent_dim,
        input_shape=config.input_shape, 
        encoder_filters=config.encoder_filters,
        encoder_kernels=config.encoder_kernels,
        decoder_filters=config.decoder_filters,
        decoder_kernels=config.decoder_kernels,
        dense_layer_units=config.dense_layer_units,
      )
      for loss_name in ["loss", "mse", "kld"]:
        metrics["val "+loss_name][epoch] = compute_av(
          metrics["val "+loss_name][epoch], 
          batch_metrics[loss_name], 
          config.steps_per_epoch_val,
        )

    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)
    logging.info('Previous minimum validation loss: {:.7f}'.format(min_val_loss))
    logging.info(
        'eval epoch: {}, train loss: {:.7f}, val loss: {:.7f}, val MSE: {:.7f}, val kld: {:.7f}'.format(
            epoch + 1,
            metrics["train loss"][epoch],
            metrics["val loss"][epoch],
            metrics["val mse"][epoch], 
            metrics["val kld"][epoch],
        )
    )
    stop = time.time()
    logging.info('Duration {:.5f}'.format(stop-start))
    if metrics["val loss"][epoch] < min_val_loss:
      
      min_val_loss = metrics["val loss"][epoch]
      ckpt = {'model': state, 'config': config}

      save_args = orbax_utils.save_args_from_target(ckpt)
      checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
      logging.info("Model saved at "+config.model_path)



@jax.jit
def eval_f_UNet(params, images, timestamps):
  def eval_model(unet_model):
    recon_images = unet_model((images[0], timestamps))
    mse_loss = mse_loss_fn(recon_images, images[1]).mean()
    return {"loss": mse_loss}

  return nn.apply(eval_model, UNet())({'params': params})

@jax.jit
def train_step_UNet(state, batch, timestamps):
  def loss_fn(params):
    recon_x = UNet().apply(
        {'params': params}, (batch[0], timestamps)
    )

    mse_loss = mse_loss_fn(recon_x, batch[1]).mean()

    return mse_loss

  loss, grads = jax.value_and_grad(loss_fn, argnums=0)(state.params)
  return state.apply_gradients(grads=grads), jnp.mean(loss)

def train_and_evaluate_UNet(
    train_tfds,
    val_tfds,
    config: ml_collections.ConfigDict,
  ):
  """Train and evaulate pipeline."""

  @jax.jit
  def compute_av(cumulative, current, num_of_steps):
    return cumulative + current / num_of_steps
  rng = random.key(0)
  rng, key = random.split(rng)

  
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
    config.model_path, orbax_checkpointer, options)

  logging.info('Initializing model.')
  init_data = (
    jnp.ones((config.batch_size, 45, 45, 6), jnp.float32), 
    jnp.ones((config.batch_size), jnp.float32),
  )
  params = UNet().init(key, init_data)['params']

  state = train_state.TrainState.create(
    apply_fn=UNet().apply,
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

  logging.info("start training...")

  for epoch in range(config.num_epochs):
    train_ds = train_tfds.as_numpy_iterator()
    # run over training steps
    train_loss = []
    for _ in range(config.steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
      timestamps = random.randint(
        key, 
        shape=(batch[0].shape[0],), 
        minval=0, 
        maxval=config.timesteps,
      )
    
      # Generating the noise and noisy image for this batch
      rng, key = random.split(rng)
      noisy_images, noise = forward_noising(key, batch[1], timestamps)
      state, batch_train_loss = train_step_UNet(state, (noisy_images, batch[1]), timestamps)
      train_loss.append(batch_train_loss)

    # run over validation steps
    metrics = {"val loss": 0.0} 
    for _ in range(config.steps_per_epoch_val):
      val_ds = val_tfds.as_numpy_iterator()
      rng, key = random.split(rng)
      timestamps = random.randint(
        key, 
        shape=(batch[0].shape[0],), 
        minval=0, 
        maxval=config.timesteps,
      )
    
      # Generating the noise and noisy image for this batch
      rng, key = random.split(rng)
      noisy_images, noise = forward_noising(key, batch[1], timestamps)
      batch_metrics = eval_f_UNet(
          state.params, 
          (noisy_images, batch[1]),
          timestamps=timestamps,
      )
      metrics["val loss"] = compute_av(
        metrics["val loss"], 
        batch_metrics["loss"], 
        config.steps_per_epoch_val,
      )

    metrics["train loss"] = np.mean(train_loss)
    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)
    logging.info('Previous minimum validation loss; {:.5f}'.format(min_val_loss))
    logging.info(
        'eval epoch: {}, train loss: {:.4f}, val loss: {:.4f}'.format(
            epoch + 1,
            metrics["train loss"],
            metrics["val loss"],
        )
    )

    if metrics["val loss"]<min_val_loss:
      logging.info("Saving model at "+config.model_path)
      min_val_loss = metrics["val loss"]
      ckpt = {'model': state, 'config': config}

      save_args = orbax_utils.save_args_from_target(ckpt)
      checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
