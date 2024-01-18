from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm

from diffdeb.diffusion import score_fn


@partial(jax.jit, static_argnames=["time_shape"])
def SED_step(step_rng, params, x, g, time_shape, time_step, step_size):
    batch_time_step = jnp.ones(time_shape) * time_step
    mean_x = x + (g**2) * score_fn(params, x, batch_time_step) * step_size

    x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)
    return x, mean_x


# pmap_score_fn = jax.pmap(score_fn, static_broadcasted_argnums=(0, 1))
def Euler_Maruyama_sampler(
    rng,
    params,
    marginal_prob_std,
    diffusion_coeff,
    start_time,
    initial_rep,
    batch_size,
    num_steps,
    exp_constant,
    eps=1e-4,
):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    rng: A JAX random state.
    score_model: A `flax.linen.Module` object that represents the architecture
      of a score-based model.
    params: A dictionary that contains the model parameters.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.
    """
    rng, step_rng = jax.random.split(rng)
    time_shape = batch_size // jax.local_device_count()
    # sample_shape = (batch_size, 12, 12, 1)
    # init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(t=.1, exp_constant=exp_constant)
    time_steps = jnp.linspace(start_time, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = initial_rep
    for time_step in tqdm(time_steps):
        g = diffusion_coeff(t=time_step, exp_constant=exp_constant)
        rng, step_rng = jax.random.split(rng)
        x, mean_x = SED_step(
            step_rng=step_rng,
            params=params,
            x=x,
            g=g,
            time_shape=time_shape,
            time_step=time_step,
            step_size=step_size,
        )
    # Do not include any noise in the last sampling step.
    return mean_x
