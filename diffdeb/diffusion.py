from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random

from diffdeb.config import get_config_diffusion
from diffdeb.models import Decoder

config = get_config_diffusion()


def compute_betas(timesteps, s=0.008):
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
branch_condt = jnp.ones(500)

branch_condt = jnp.ones(500)
branch_condt = branch_condt.at[0].set(0)


# Implement noising logic according to reparameterization trick
@jax.jit
def forward_noising(key, x_0, t):
    noise = random.normal(key, x_0.shape)

    reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sqrt_alpha_bar, t), (-1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(
        jnp.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1)
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
def backward_denoising_ddpm(input_key, x_t, pred_noise, t):
    alpha_t = jnp.take(alpha, t)
    alpha_t_bar = jnp.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5
    mean = 1 / (alpha_t**0.5) * (x_t - eps_coef * pred_noise)

    var = jnp.take(beta, t)
    z = random.normal(input_key, shape=x_t.shape)

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


@partial(
    jax.jit,
    static_argnames=[
        "input_shape",
        "latent_dim",
        "decoder_filters",
        "decoder_kernels",
        "dense_layer_units",
        "padding_infos",
    ],
)
def inverse_prob_ddpm_step(
    input_key,
    x_t,
    pred_noise,
    t,
    y,
    latent_scaling_factor,
    noise_sigma,
    decoder_params,
    input_shape,
    latent_dim,
    decoder_filters,
    decoder_kernels,
    dense_layer_units,
    padding_infos,
):

    # latent_shape = x_t.shape

    alpha_t_bar = jnp.take(alpha_bar, t)
    # alpha_t_m1_bar = jnp.take(alpha_bar, t - 1)
    alpha_t = jnp.take(alpha, t)
    # beta_t = jnp.take(beta, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5
    mean = 1 / (alpha_t**0.5) * (x_t - eps_coef * pred_noise)

    var = jnp.take(beta, t)
    z = random.normal(input_key, shape=x_t.shape)
    x_t_m1_bar = mean + (var**0.5) * z * jnp.take(branch_condt, t - 1)

    # x_0 =  (x_t - (1-alpha_t_bar) * pred_noise) / (alpha_t_bar ** 0.5)
    # # #x_0 = jnp.clip(x_0, -1, +1)
    # var = jnp.take(beta, t)
    # z = random.normal(input_key, shape=x_t.shape)

    # # print(var ** 0.5)

    # x_t_m1_bar = (alpha_t ** 0.5) *  ((1 - alpha_t_m1_bar)/(1 - alpha_t_bar)) * x_t
    # + (alpha_t_m1_bar ** 0.5) * beta_t / (1 - alpha_t_bar) * x_0
    # + (var ** 0.5) * z

    def gauss_likelihood_fn(x_t, pred_noise):
        x_0 = (x_t - (1 - alpha_t_bar) ** 0.5 * pred_noise) / (alpha_t_bar**0.5)
        reconst = Decoder(
            latent_dim=latent_dim,
            input_shape=input_shape,
            filters=decoder_filters,
            kernels=decoder_kernels,
            dense_layer_units=dense_layer_units,
        ).apply({"params": decoder_params}, x_0[:, :, 0])

        reconst_field = jnp.pad(reconst[0], padding_infos[0])
        for i in range(len(padding_infos) - 1):
            reconst_field += jnp.pad(reconst[i + 1], padding_infos[i + 1])

        absolute_difference = jnp.abs(y - reconst_field)

        mse = (absolute_difference) ** 2
        var = noise_sigma**2

        gauss_likelihood = mse / (2 * var)

        gauss_likelihood = mse
        return jnp.sum(gauss_likelihood), jnp.linalg.norm(absolute_difference)

    gradient, absolute_difference = jax.vmap(
        jax.grad(gauss_likelihood_fn, has_aux=True)
    )(x_t, pred_noise)

    x_t_m1 = x_t_m1_bar - 0.005 * gradient / absolute_difference.reshape(-1, 1, 1, 1)
    # x_t_m1 = (
    #     x_t_m1_bar - 0.00001 *gradient / absolute_difference.reshape(-1, 1, 1, 1)
    # )
    # x_t_m1 = x_t_m1_bar

    return x_t_m1, x_t_m1_bar, 0, (var**0.5) * z


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
