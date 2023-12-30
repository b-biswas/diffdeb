from typing import Sequence
from flax import linen as nn
from jax import random
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

class SinusoidalEmbedding(nn.Module):
  dim: int = 32
  
  @nn.compact
  def __call__(self, inputs):
    half_dim = self.dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = inputs[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
    return emb


class TimeEmbedding(nn.Module):
  dim: int = 32
  @nn.compact
  def __call__(self, inputs):
    time_dim = self.dim * 4
    
    se = SinusoidalEmbedding(self.dim)(inputs)
    
    # Projecting the embedding into a 128 dimensional space
    x = nn.Dense(time_dim)(se)
    x = nn.gelu(x)
    x = nn.Dense(time_dim)(x)
    
    return x

# Standard dot-product attention with eight heads.
class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        batch, h, w, channels = inputs.shape
        inputs = inputs.reshape(batch, h*w, channels)
        batch, n, channels = inputs.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(
            self.dim * 3, use_bias=self.use_bias, kernel_init=self.kernel_init
        )(inputs)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attention = nn.softmax(attention, axis=-1)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = jnp.reshape(x, (batch, int(x.shape[1]** 0.5), int(x.shape[1]** 0.5), -1))
        return x
    
class Block(nn.Module):
  dim: int = 32
  groups: int = 8

  @nn.compact
  def __call__(self, inputs):
    conv = nn.Conv(self.dim, (3, 3))(inputs)
    norm = nn.GroupNorm(num_groups=self.groups)(conv)
    activation = nn.silu(norm)
    return activation


class ResnetBlock(nn.Module):
  dim: int = 32
  groups: int = 8

  @nn.compact
  def __call__(self, inputs, time_embed=None):
    x = Block(self.dim, self.groups)(inputs)
    if time_embed is not None:
      time_embed = nn.silu(time_embed)
      time_embed = nn.Dense(self.dim)(time_embed)
      print(time_embed.shape)
      x = jnp.expand_dims(jnp.expand_dims(time_embed, 1), 1) + x
      print(x.shape)
    x = Block(self.dim, self.groups)(x)
    res_conv = nn.Conv(self.dim, (1, 1), padding="SAME")(inputs)
    return x + res_conv
  
class UNet(nn.Module):
  dim: int = 8 # controls the number of channels in the UNet layers
  dim_scale_factor: tuple = (1, 2, 4, 8)
  num_groups: int = 8

  @nn.compact
  def __call__(self, inputs):
    inputs, time = inputs
    channels = inputs.shape[-1]

    # pad inputs so that stamp size is divisible by 2^num of conv stride 2 layers
    # otherwise dimentions won't match while concatenating the UNet layers
    original_stamp_size = inputs.shape[1]
    factor_of_reduction = 2**len(self.dim_scale_factor)
    padding_to_add = factor_of_reduction - inputs.shape[1]  % factor_of_reduction
    if padding_to_add != 0:
      inputs = jnp.pad(
        inputs,
        pad_width=(
          (0,0), 
          (0, padding_to_add),
          (0, padding_to_add),
          (0,0),
        ),
      )
    print(inputs.shape)
    x = nn.Conv(self.dim // 3 * 2, (5, 5), padding="SAME")(inputs)
    time_emb = TimeEmbedding(self.dim)(time)
    
    dims = [self.dim * i for i in self.dim_scale_factor]
    pre_downsampling = []
    print(x.shape)
    # Downsampling phase
    for index, dim in enumerate(dims):
      x = ResnetBlock(dim, self.num_groups)(x, time_emb)
      print(x.shape)
      x = ResnetBlock(dim, self.num_groups)(x, time_emb)
      print(x.shape)
      att = Attention(dim)(x)
      norm = nn.GroupNorm(self.num_groups)(att)
      x = norm + x
      # Saving this output for residual connection with the upsampling layer
      pre_downsampling.append(x)
      if index != len(dims) - 1:
        x = nn.Conv(dim, (5,5), (2,2))(x)
    
    # Middle block
    x = ResnetBlock(dims[-1], self.num_groups)(x, time_emb)
    att = Attention(dim)(x)
    norm = nn.GroupNorm(self.num_groups)(att)
    x = norm + x 
    x = ResnetBlock(dims[-1], self.num_groups)(x, time_emb)
    
    # Upsampling phase
    for index, dim in enumerate(reversed(dims)):
      x = jnp.concatenate([pre_downsampling.pop(), x], -1)
      x = ResnetBlock(dim, self.num_groups)(x, time_emb)
      x = ResnetBlock(dim, self.num_groups)(x, time_emb)
      att = Attention(dim)(x)
      norm = nn.GroupNorm(self.num_groups)(att)
      x = norm + x
      if index != len(dims) - 1:
        x = nn.ConvTranspose(dim, (5,5), (2,2))(x)


    # Final ResNet block and output convolutional layer
    x = ResnetBlock(dim, self.num_groups)(x, time_emb)
    x = nn.Conv(channels, (1,1), padding="SAME")(x)

    x = x[:, 0:original_stamp_size, 0:original_stamp_size, :]
    return x

def create_UNet_model():
  return UNet()