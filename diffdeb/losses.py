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