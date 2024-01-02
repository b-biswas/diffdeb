import jax 
import jax.numpy as jnp

@jax.jit
@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
@jax.vmap
def mse_loss_fn(prediction, truth):
  return (prediction-truth) ** 2

@jax.jit
def vae_train_loss(prediction, truth, mean, logvar):
  mse_loss = mse_loss_fn(prediction, truth).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  loss = mse_loss + kld_loss

  return loss