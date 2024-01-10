import jax 
import jax.numpy as jnp

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def mse_loss_fn(prediction, truth):
  return jnp.sum(((prediction-truth) ** 2))

@jax.jit
def vae_train_loss(prediction, truth, mean, logvar, kl_weight):
  mse_loss = mse_loss_fn(prediction, truth).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  loss = mse_loss + kl_weight*kld_loss
  #loss = mse_loss
  return loss , mse_loss, kld_loss