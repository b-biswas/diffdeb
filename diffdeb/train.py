from diffdeb.losses import kl_divergence, mse_loss_fn
from diffdeb.models import create_vae_model, UNet
import jax 
import jax.numpy as jnp
from jax import random 
import logging

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
    num_epochs,
    steps_per_epoch_train, 
    steps_per_epoch_val,
    batch_size=32,
    latent_dim=16,
    learning_rate=1e-4,
  ):
  """Train and evaulate pipeline."""
  rng = random.key(0)
  rng, key = random.split(rng)

  logging.info('Initializing model.')
  init_data = jnp.ones((batch_size, 45, 45, 6), jnp.float32)
  params = create_vae_model(latent_dim=16).init(key, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=create_vae_model(latent_dim=16).apply,
      params=params,
      tx=optax.adam(learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (32, latent_dim))

  for epoch in range(num_epochs):
    for _ in range(steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step_vae(state, batch, key, latent_dim)

    metrics, comparison, sample = eval_f_vae(
        state.params, next(val_ds), z, eval_rng, latent_dim,
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

def eval_f_UNet(params, images):
  def eval_model(unet_model):
    recon_images = unet_model(images[0])

    mse_loss = mse_loss_fn(recon_images, images[1])
    return {'mse': mse_loss, 'loss': mse_loss}

  return nn.apply(eval_model, UNet())({'params': params})

def train_step_UNet(state, batch):
  def loss_fn(params):
    recon_x = UNet.apply(
        {'params': params}, batch[0]
    )

    mse_loss = mse_loss_fn(recon_x, batch[1]).mean()

    return mse_loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

def train_and_evaluate_UNet(
    train_ds,
    val_ds,
    num_epochs,
    steps_per_epoch_train, 
    batch_size=32,
    learning_rate=1e-4,
  ):
  """Train and evaulate pipeline."""
  rng = random.key(0)
  rng, key = random.split(rng)

  logging.info('Initializing model.')
  init_data = jnp.ones((batch_size, 45, 45, 6), jnp.float32)
  params = UNet().init(key, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=UNet().apply,
      params=params,
      tx=optax.adam(learning_rate),
  )


  for epoch in range(num_epochs):
    for _ in range(steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step_UNet(state, batch, key)

    metrics = eval_f_UNet(
        state.params, next(val_ds)
    )
    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print(
        'eval epoch: {}, loss: {:.4f}, MSE: {:.4f}'.format(
            epoch + 1,
            metrics['loss'],
            metrics['mse'], 
        )
    )