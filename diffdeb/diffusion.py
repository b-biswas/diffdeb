from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random

from diffdeb.models import UNet

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


@jax.jit
def marginal_prob_std(t, exp_constant):
    r"""Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    return jnp.sqrt((exp_constant ** (2 * t) - 1.0) / 2.0 / jnp.log(exp_constant))


@jax.jit
def diffusion_coeff(t, exp_constant):
    r"""Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      exp_constant: The exp_constant in the SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return exp_constant**t


@jax.jit
def forward_SED_noising(key, x_0, t, exp_constant):
    normal_noise = random.normal(
        key, x_0.shape
    )  # This noise is probably too much for CATSIM dataset
    std = marginal_prob_std(
        t=t,
        exp_constant=exp_constant,
    )
    std = std.reshape(-1, 1, 1, 1)
    noise = std * normal_noise
    noisy_image = x_0 + noise
    return noisy_image, noise, std


@jax.jit
def score_fn(params, x, t):
    return UNet().apply({"params": params}, (x, t))
