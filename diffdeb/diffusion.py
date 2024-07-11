import random as r

import jax
import jax.numpy as jnp
import jax.random as random

from diffdeb.config import get_config_diffusion

config = get_config_diffusion()


def compute_betas(timesteps, s=0.00008):
    def f(t):
        return jnp.cos((t / timesteps + s) / (1 + s) * 0.5 * jnp.pi) ** 2

    x = jnp.linspace(0, timesteps, timesteps + 1)
    alpha_bar = f(x) / f(jnp.asarray([0]))
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return jnp.clip(betas, 0.0001, 0.05)


# Defining beta for all t's in T steps
beta = compute_betas(timesteps=config.timesteps)

# Defining alpha and its derivatives according to reparameterization trick
alpha = 1 - beta
alpha_bar = jnp.cumprod(alpha, 0)
alpha_bar = jnp.concatenate((jnp.array([1.0]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = jnp.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = jnp.sqrt(1 - alpha_bar)


# Implement noising logic according to reparameterization trick
@jax.jit
def forward_noising(key, x_0, t):
    noise = random.normal(key, x_0.shape)

    reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(
        jnp.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1)
    )

    noisy_image = (
        reshaped_sqrt_alpha_bar_t * x_0 + reshaped_one_minus_sqrt_alpha_bar_t * noise
    )
    return noisy_image, noise


# This function defines the logic of getting x_t-1 given x_t
# @partial(
#     jax.jit,
#     static_argnames=["image_shape"],
# )
@jax.jit
def backward_denoising_ddpm(x_t, pred_noise, t):
    alpha_t = jnp.take(alpha, t)
    alpha_t_bar = jnp.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5
    mean = 1 / (alpha_t**0.5) * (x_t - eps_coef * pred_noise)

    var = jnp.take(beta, t)
    z = random.normal(key=random.PRNGKey(r.randint(1, 1000)), shape=x_t.shape)

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


# @partial(
#     jax.jit,
#     static_argnames=["input_shape", "use_likelihood", "latent_dim", "decoder_filters", "decoder_kernels", "dense_layer_units"],
# )
# def score_fn(params, x, t,
#     use_likelihood=False,
#     y=0,
#     decoder_params=0,
#     input_shape=0,
#     latent_dim=0,
#     decoder_filters=0,
#     decoder_kernels=0,
#     dense_layer_units=0,
#     ):
#     score = UNet().apply({"params": params}, (x, t))
#     if use_likelihood:
#         def gauss_likelihood_fn(x):
#             gal_img = Decoder(
#                 latent_dim=latent_dim,
#                 input_shape=input_shape,
#                 filters=decoder_filters,
#                 kernels=decoder_kernels,
#                 dense_layer_units=dense_layer_units,
#             ).apply({"params":decoder_params}, x)
#             mse = (y - gal_img)**2
#             #var = sigma**2
#             #gauss_likelihood = mse/(2*var)
#             gauss_likelihood = mse
#             return gauss_likelihood
#         score = jax.grad(gauss_likelihood_fn)(x) + score
#     return  score


# def backward_denoising_ddim(x_t, pred_noise, t, sigma_t):
#   alpha_bar_t = jnp.take(alpha_bar, t)
#   alpha_t_minus_one = jnp.take(alpha, t - 1)

#   # predicted x_0
#   pred = (x_t - ((1 - alpha_bar_t) ** 0.5) * pred_noise)/ (alpha_bar_t ** 0.5)
#   pred = (alpha_t_minus_one ** 0.5) * pred

#   # direction pointing to x_t
#   pred = pred + ((1 - alpha_t_minus_one - (sigma_t ** 2)) ** 0.5) * pred_noise

#   # random noise
#   eps_t = random.normal(key=random.PRNGKey(r.randint(1, 100)), shape=x_t.shape)
#   pred = pred + (sigma_t * eps_t)

#   return pred
