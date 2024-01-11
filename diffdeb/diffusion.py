from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random

# Defining a constant value for T
timesteps = 200

# Defining beta for all t's in T steps
beta = jnp.linspace(0.0001, 0.02, timesteps)

# Defining alpha and its derivatives according to reparameterization trick
alpha = 1 - beta
alpha_bar = jnp.cumprod(alpha, 0)
alpha_bar = jnp.concatenate((jnp.array([1.0]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = jnp.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = jnp.sqrt(1 - alpha_bar)


# Implement noising logic according to reparameterization trick
@jax.jit
def forward_noising(key, x_0, t):
    noise = random.normal(
        key, x_0.shape
    )  # This noise is probably too much for CATSIM dataset
    reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(
        jnp.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1)
    )
    noisy_image = (
        reshaped_sqrt_alpha_bar_t * x_0 + reshaped_one_minus_sqrt_alpha_bar_t * noise
    )
    return noisy_image, noise


# This function defines the logic of getting x_t-1 given x_t
@partial(
    jax.jit,
    static_argnames=["image_shape"],
)
def backward_denoising_ddpm(x_t, pred_noise, t, image_shape, rng):
    alpha_t = jnp.take(alpha, t)
    alpha_t_bar = jnp.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5
    mean = 1 / (alpha_t**0.5) * (x_t - eps_coef * pred_noise)

    var = jnp.take(beta, t)
    z = random.normal(key=rng, shape=image_shape)

    return mean + (var**0.5) * z
