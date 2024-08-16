import jax
import jax.numpy as jnp


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar / 2 - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def mse_loss_fn(prediction, truth):
    return jnp.sum((prediction - truth) ** 2)


@jax.vmap
def scaled_mse_loss_fn(prediction, truth):
    return (prediction - truth) ** 2


@jax.jit
def reconst_loss_fn(prediction, truth, noise_sigma, linear_norm_coeff):
    dinominator = truth / linear_norm_coeff + noise_sigma**2
    mse = (prediction - truth) ** 2
    output = mse / dinominator
    return output.sum()


@jax.jit
def vae_train_loss(
    prediction, truth, mean, logvar, kl_weight, noise_sigma, linear_norm_coeff
):
    vmapped_loss = jax.vmap(
        reconst_loss_fn, (0, 0, None, None), 0
    )  # TODO: where to put this line?
    recon_loss = vmapped_loss(prediction, truth, noise_sigma, linear_norm_coeff).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = recon_loss + kl_weight * kld_loss
    # loss = mse_loss
    return loss, recon_loss, kld_loss
