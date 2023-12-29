from typing import Sequence
from flax import linen as nn
from jax import random, math
import jax.numpy as jnp

from collections.abc import Callable

class Encoder(nn.Module):
  """VAE Encoder."""

  latent_dim: int
  filters: Sequence[int]
  kernels: Sequence[int]
  dense_layer_units: int

  # def __init__(self, latent_dim, filters, kernels, dense_layer_units):
  #    self.latent_dim = latent_dim
  #    self.filters = filters
  #    self.kernels = kernels
  #    self.dense_layer_units = dense_layer_units

  @nn.compact
  def __call__(self, x):

    for i in range(len(self.filters)):
        x = nn.Conv(
            features=self.filters[i],
            kernel_size=(self.kernels[i], self.kernels[i]),
            padding="SAME",
            strides=(2, 2),
        )(x)
        x = nn.activation.PReLU()(x)

    x = x.reshape(x.shape[0], -1)
    x = nn.Dense(features=self.dense_layer_units)(x)
    x = nn.activation.PReLU()(x)

    mean_x = nn.Dense(features=self.latent_dim, name='latent_mean')(x)
    logvar_x = nn.Dense(features=self.latent_dim, name='latent_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """VAE Decoder."""

  input_shape: Sequence[int]
  latent_dim: int
  filters: Sequence[int]
  kernels: Sequence[int]
  dense_layer_units: int

  # def __init__(self, latent_dim, filters, kernels, dense_layer_units, input_shape):
  #    self.latent_dim = latent_dim
  #    self.filters = filters
  #    self.kernels = kernels
  #    self.dense_layer_units = dense_layer_units
  #    self.input_shape = input_shape

  @nn.compact
  def __call__(self, z):
    # z = nn.Dense(500, name='fc1')(z)
    # z = nn.relu(z)
    # z = nn.Dense(784, name='fc2')(z)

    z = nn.Dense(features=self.dense_layer_units)(z)
    z = nn.activation.PReLU()(z)
    w = int(jnp.ceil(self.input_shape[0] / 2 ** (len(self.filters))))
    z = nn.Dense(features=w * w * self.filters[-1])(z)
    z = nn.activation.PReLU()(z)
    z = z.reshape(z.shape[0], w, w, self.filters[-1])
    for i in range(len(self.filters) - 1, -1, -1):
        z = nn.ConvTranspose(
            features=self.filters[i],
            kernel_size=(self.kernels[i], self.kernels[i]),
            padding="SAME",
            strides=(2, 2),
        )(z)
        z = nn.activation.PReLU()(z)

    z = nn.ConvTranspose(
        features=self.filters[0],
        kernel_size=(3, 3),
        padding="SAME",
    )(z)
    z = nn.activation.PReLU()(z)

    # keep the output of the last layer as relu as we want only positive flux values.
    z = nn.ConvTranspose(self.input_shape[-1], (3, 3), padding="SAME")(z)
    z = nn.activation.relu(z)
    print(z.shape)

    # In case the last convolutional layer does not provide an image of the size of the input image, cropp it.
    cropping = z.shape[1] - self.input_shape[0]
    if cropping > 0:
        z=z[:, 0:self.input_shape[0], 0:self.input_shape[1]]
    print(z.shape)
    return z


class VAE(nn.Module):
  """Full VAE model."""

  input_shape: Sequence[int]
  latent_dim: int
  filters: Sequence[int]
  kernels: Sequence[int]
  dense_layer_units: int

  # def __init__(self, latent_dim, filters, kernels, dense_layer_units, input_shape):
  #    super().__init__()
  #    self.latent_dim = latent_dim
  #    self.filters = filters
  #    self.kernels = kernels
  #    self.dense_layer_units = dense_layer_units
  #    self.input_shape = input_shape

  def setup(self):
    self.encoder = Encoder(
       latent_dim=self.latent_dim, 
       filters=self.filters, 
       kernels=self.kernels, 
       dense_layer_units=self.dense_layer_units,
      )
    self.decoder = Decoder(
       latent_dim=self.latent_dim, 
       filters=self.filters, 
       kernels=self.kernels, 
       dense_layer_units=self.dense_layer_units, 
       input_shape=self.input_shape,
      )

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar
  
  def generate(self, z):
    return self.decoder(z)

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

def create_VAE_model(latent_dim,
    filters=[15, 15, 15, 15], 
    kernels=[3, 3, 3, 3], 
    dense_layer_units=120, 
    input_shape=[45, 45, 6],
  ):
  return VAE(
    latent_dim=latent_dim, 
    filters=filters, 
    kernels=kernels, 
    dense_layer_units=dense_layer_units, 
    input_shape=input_shape,
  )

