from diffdeb.losses import kl_divergence, mse_loss_fn
from diffdeb.models import create_vae_model, UNet
import jax 
import jax.numpy as jnp
import numpy as np
from jax import random 
import logging
from diffdeb.diff_utils import forward_noising

import ml_collections

import flax.linen as nn
from flax.training import train_state

import optax

def compute_metrics_vae(recon_x, x, mean, logvar):
  mse_loss = mse_loss_fn(prediction=recon_x, truth=x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'mse': mse_loss, 'kld': kld_loss, 'loss': mse_loss + kld_loss}

def train_step_vae(state, batch, z_rng, latent_dim):
  def loss_fn(params):
    recon_x, mean, logvar = create_vae_model(latent_dim).apply(
        {'params': params}, batch[0], z_rng
    )

    mse_loss = mse_loss_fn(recon_x, batch[1]).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = mse_loss + kld_loss
    return loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

@jax.jit
def eval_f_vae(params, images, z, z_rng, latent_dim):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images[0], z_rng)
    comparison = jnp.concatenate([
        images[1].reshape(-1, 45, 45, 6),
        recon_images.reshape(-1, 45, 45, 6),
    ])

    generate_images = vae.generate(z)
    generate_images = generate_images.reshape(-1, 45, 45, 6)
    metrics = compute_metrics_vae(recon_images, images[1], mean, logvar)
    return metrics, comparison, generate_images

  return nn.apply(eval_model, create_vae_model(latent_dim))({'params': params})

def train_and_evaluate_vae(
    train_ds,
    val_ds,
    config: ml_collections.ConfigDict,
  ):
  """Train and evaulate pipeline."""
  rng = random.key(0)
  rng, key = random.split(rng)

  logging.info('Initializing model.')
  init_data = jnp.ones((config.batch_size, 45, 45, 6), jnp.float32)
  params = create_vae_model(latent_dim=16).init(key, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=create_vae_model(latent_dim=16).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (32, config.latent_dim))

  for epoch in range(config.num_epochs):
    for _ in range(config.steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step_vae(state, batch, key, config.latent_dim)

    metrics, comparison, sample = eval_f_vae(
        state.params, next(val_ds), z, eval_rng, config.latent_dim,
    )
    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print(
        'eval epoch: {}, loss: {:.4f}, MSE: {:.4f}, KLD: {:.4f}'.format(
            epoch + 1,
            metrics['loss'], 
            metrics['mse'], 
            metrics['kld'],
        )
    )

def eval_f_UNet(params, images, timestamps):
  def eval_model(unet_model):
    recon_images = unet_model((images[0], timestamps))

    mse_loss = mse_loss_fn(recon_images, images[1])
    return {'mse': mse_loss, 'loss': mse_loss}

  return nn.apply(eval_model, UNet())({'params': params})

def train_step_UNet(state, batch, timestamps):
  def loss_fn(params):
    recon_x = UNet().apply(
        {'params': params}, (batch[0], timestamps)
    )

    mse_loss = mse_loss_fn(recon_x, batch[1]).mean()

    return mse_loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

def train_and_evaluate_UNet(
    train_ds,
    val_ds,
    config: ml_collections.ConfigDict,
  ):
  """Train and evaulate pipeline."""
  rng = random.key(0)
  rng, key = random.split(rng)

  logging.info('Initializing model.')
  init_data = (
    jnp.ones((config.batch_size, 45, 45, 6), jnp.float32), 
    jnp.ones((config.batch_size), jnp.float32),
  )
  params = UNet().init(key, init_data)['params']

  state = train_state.TrainState.create(
      apply_fn=UNet().apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )


  for epoch in range(config.num_epochs):
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
    state = train_step_UNet(state, (noisy_images, batch[1]), timestamps)
    val_loss = []
    val_mse = []

    # run over validation steps
    for _ in range(config.steps_per_epoch_val):
      batch = next(val_ds)
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
    metrics = eval_f_UNet(
        state.params, 
        (noisy_images, batch[1]),
        timestamps=timestamps,
    )
    val_loss.append(metrics['loss'])
    val_mse.append(metrics['mse'])
    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print(
        'eval epoch: {}, loss: {:.4f}, MSE: {:.4f}'.format(
            epoch + 1,
            np.mean(val_loss),
            np.mean(val_mse), 
        )
    )