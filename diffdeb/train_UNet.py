import os 
import shutil 
import time
import logging

from functools import partial
import ml_collections

import numpy as np
import jax 
import jax.numpy as jnp
from jax import random 

import flax.linen as nn
from flax.training import train_state
from flax.training import orbax_utils
import orbax
import optax

from diffdeb.diff_utils import forward_noising
from diffdeb.losses import mse_loss_fn
from diffdeb.models import UNet

logging.basicConfig(level=logging.INFO)

@jax.jit
def eval_f_UNet(params, images, timestamps):
  def eval_model(unet_model):
    recon_images = unet_model((images[0], timestamps))
    mse_loss = mse_loss_fn(recon_images, images[1]).mean()
    return {"loss": mse_loss}

  return nn.apply(eval_model, UNet())({'params': params})

@jax.jit
def train_step_UNet(state, batch, timestamps):
  def loss_fn(params):
    recon_x = UNet().apply(
        {'params': params}, (batch[0], timestamps)
    )

    mse_loss = mse_loss_fn(recon_x, batch[1]).mean()

    return mse_loss

  loss, grads = jax.value_and_grad(loss_fn, argnums=0)(state.params)
  return state.apply_gradients(grads=grads), loss

def train_and_evaluate_UNet(
    train_tfds,
    val_tfds,
    config: ml_collections.ConfigDict,
  ):
  """Train and evaulate pipeline."""

  @jax.jit
  def compute_av(cumulative, current, num_of_steps):
    return cumulative + current / num_of_steps
  rng = random.key(0)
  rng, key = random.split(rng)

  # Define checkpoint
  if os.path.exists(config.model_path):
    shutil.rmtree(config.model_path)

  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
    config.model_path, orbax_checkpointer, options)

  logging.info('Initializing model.')
  init_data = (
    jnp.ones((config.batch_size, 45, 45, 6), jnp.float32), 
    jnp.ones((config.batch_size), jnp.float32),
  )
  params = UNet().init(key, init_data)['params']

  state = train_state.TrainState.create(
    apply_fn=UNet().apply,
    params=params,
    tx=optax.chain(
      optax.clip(.1),
      optax.adam(
        learning_rate=optax.exponential_decay(
          config.learning_rate,
          transition_steps=config.steps_per_epoch_train*30,
          decay_rate=.1,
          end_value=1e-7,
        ),
      ),
    ),
  )
      

  min_val_loss = np.inf

  logging.info("start training...")
  
  metrics = {"train loss": [], "val loss":[]} 
  for epoch in range(config.num_epochs):
    start = time.time()
    train_ds = train_tfds.as_numpy_iterator()
    # run over training steps
    
    current_epoch_train_loss = 0
    for _ in range(config.steps_per_epoch_train):
      batch = next(train_ds)
      rng, key = random.split(rng)
      timestamps = random.randint(
        key, 
        shape=(batch[0].shape[0],), 
        minval=0, 
        maxval=config.timesteps,
      )
    
      # Generating the noise and noisy image for this batch
      rng, key = random.split(rng)
      noisy_images, noise = forward_noising(key, batch[1], timestamps)
      state, batch_train_loss = train_step_UNet(state, (noisy_images, batch[1]), timestamps)
      current_epoch_train_loss = compute_av(
        current_epoch_train_loss, 
        batch_train_loss, 
        config.steps_per_epoch_train,
      )
      metrics["train loss"].append(current_epoch_train_loss)

    # run over validation steps
    metrics["val loss"].append(0.0)
    for _ in range(config.steps_per_epoch_val):
      val_ds = val_tfds.as_numpy_iterator()
      batch = next(val_ds)
      rng, key = random.split(rng)
      timestamps = random.randint(
        key, 
        shape=(batch[0].shape[0],), 
        minval=0, 
        maxval=config.timesteps,
      )
    
      # Generating the noise and noisy image for this batch
      rng, key = random.split(rng)
      noisy_images, noise = forward_noising(key, batch[1], timestamps)
      batch_metrics = eval_f_UNet(
          state.params, 
          (noisy_images, batch[1]),
          timestamps=timestamps,
      )
      metrics["val loss"][epoch] = compute_av(
        metrics["val loss"][epoch], 
        batch_metrics["loss"], 
        config.steps_per_epoch_val,
      )
    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8
    # )
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)
    #logging.info('Previous minimum validation loss; {:.5f}'.format(min_val_loss))
    logging.info(
        '\n\neval epoch: {}, train loss: {:.4f}, val loss: {:.4f}'.format(
            epoch + 1,
            metrics["train loss"][epoch],
            metrics["val loss"][epoch],
        )
    )
    end = time.time()
    logging.info("\nTotal time taken: {} seconds".format(end - start))
    if metrics["val loss"][epoch]<min_val_loss:
      logging.info("\nVal loss improved from: {:.7f} to {:.7f}".format(min_val_loss, metrics["val loss"][epoch]))

      min_val_loss = metrics["val loss"][epoch]
      ckpt = {'model': state, 'config': config, 'metrics': metrics}

      save_args = orbax_utils.save_args_from_target(ckpt)
      checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
