from collections.abc import Callable
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random


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
                strides=(1, 1),
            )(x)
            x = nn.activation.PReLU()(x)

            x = nn.Conv(
                features=self.filters[i],
                kernel_size=(self.kernels[i], self.kernels[i]),
                padding="SAME",
                strides=(2, 2),
            )(x)
            x = nn.activation.PReLU()(x)

        if self.dense_layer_units != 0:
            x = x.reshape(x.shape[0], -1)
            x = nn.Dense(features=self.dense_layer_units)(x)
            x = nn.activation.PReLU()(x)

            mean_x = nn.Dense(features=self.latent_dim, name="latent_mean")(x)
            logvar_x = nn.Dense(features=self.latent_dim, name="latent_logvar")(x)
            logvar_x = -nn.relu(logvar_x)
        else:
            mean_x1 = nn.Conv(
                features=16,
                kernel_size=(5, 5),
                padding="SAME",
            )(x)
            mean_x1 = nn.activation.PReLU()(mean_x1)
            mean_x = nn.Conv(
                features=2,
                kernel_size=(5, 5),
                padding="SAME",
            )(mean_x1)
            mean_x = nn.activation.tanh(mean_x)

            logvar_x1 = nn.Conv(
                features=16,
                kernel_size=(5, 5),
                padding="SAME",
            )(x)
            logvar_x1 = nn.activation.PReLU()(logvar_x1)
            logvar_x = nn.Conv(
                features=2,
                kernel_size=(5, 5),
                padding="SAME",
            )(logvar_x1)
            logvar_x = -nn.relu(logvar_x)

        return mean_x, logvar_x


class Decoder(nn.Module):
    """VAE Decoder."""

    input_shape: Sequence[int]
    latent_dim: int
    filters: Sequence[int]
    kernels: Sequence[int]
    dense_layer_units: int

    @nn.compact
    def __call__(self, z):

        if self.dense_layer_units != 0:
            z = nn.Dense(features=self.dense_layer_units)(z)
            z = nn.activation.PReLU()(z)
            w = int(self.input_shape[0] // 2 ** (len(self.filters)) + 1)
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
        z = nn.relu(z)

        # In case the last convolutional layer does not provide an image of the size of the input image, cropp it.
        cropping = z.shape[1] - self.input_shape[0]
        if cropping > 0:
            z = z[:, 0 : self.input_shape[0], 0 : self.input_shape[1]]
        return z


class VAE(nn.Module):
    """Full VAE model."""

    input_shape: Sequence[int]
    latent_dim: int
    encoder_filters: Sequence[int] = (32, 128, 256, 512)
    encoder_kernels: Sequence[int] = (5, 5, 5, 5)
    decoder_filters: Sequence[int] = (64, 96, 128)
    decoder_kernels: Sequence[int] = (5, 5, 5)
    dense_layer_units: int = 512

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
            filters=self.encoder_filters,
            kernels=self.encoder_kernels,
            dense_layer_units=self.dense_layer_units,
        )
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            input_shape=self.input_shape,
            filters=self.decoder_filters,
            kernels=self.decoder_kernels,
            dense_layer_units=self.dense_layer_units,
        )

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return self.decoder(z)


@jax.jit
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def create_vae_model(
    latent_dim,
    input_shape,
    encoder_filters,
    encoder_kernels,
    decoder_filters,
    decoder_kernels,
    dense_layer_units,
):
    return VAE(
        latent_dim=latent_dim,
        input_shape=input_shape,
        encoder_filters=encoder_filters,
        encoder_kernels=encoder_kernels,
        decoder_filters=decoder_filters,
        decoder_kernels=decoder_kernels,
        dense_layer_units=dense_layer_units,
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
        inputs = inputs.reshape(batch, h * w, channels)
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
        x = jnp.reshape(x, (batch, int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5), -1))
        return x


class Block(nn.Module):
    dim: int = 32

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(self.dim, [int(3)])(inputs)
        x = nn.activation.PReLU()(x)
        return x


class ResnetBlock(nn.Module):
    dim: int = 32

    @nn.compact
    def __call__(self, inputs, time_embed=None):
        x = Block(self.dim)(inputs)
        if time_embed is not None:
            # time_embed = nn.silu(time_embed)
            time_embed = nn.Dense(self.dim)(time_embed)
            time_embed = nn.activation.PReLU()(time_embed)
            x = jnp.expand_dims(time_embed, 1) + x
        x = Block(self.dim)(x)
        # res_conv = nn.Conv(self.dim, [int(3)], padding="SAME")(inputs)
        return x + inputs


class UNet(nn.Module):
    dim: int = 32  # controls the number of channels in the UNet layers
    num_down_sampling: int = 2

    @nn.compact
    def __call__(self, inputs):
        inputs, time = inputs
        channels = inputs.shape[-1]

        # pad inputs so that stamp size is divisible by 2^num of conv stride 2 layers
        # otherwise dimentions won't match while concatenating the UNet layers
        # original_stamp_size = inputs.shape[1]
        # factor_of_reduction = 2 ** len(self.dim_scale_factor)
        # padding_to_add = factor_of_reduction - inputs.shape[1] % factor_of_reduction
        # if padding_to_add != 0:
        #     inputs = jnp.pad(
        #         inputs,
        #         pad_width=(
        #             (0, 0),
        #             (0, padding_to_add),
        #             (0, padding_to_add),
        #             (0, 0),
        #         ),
        #     )
        # inputs=jnp.expand_dims(inputs, -1)
        x = nn.Conv(self.dim, [int(3)], padding="SAME")(inputs)
        time_emb = TimeEmbedding(self.dim)(time)

        pre_downsampling = []
        # Downsampling phase
        for i in range(self.num_down_sampling):
            x = ResnetBlock()(x, time_emb)
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            x = nn.Conv(self.dim, [int(3)], [int(2)])(x)

        # Middle block
        x = ResnetBlock()(x, time_emb)
        x = ResnetBlock()(x, time_emb)

        # Upsampling phase
        x = nn.ConvTranspose(self.dim, [int(3)], [int(2)])(x)
        for i in range(self.num_down_sampling - 1):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = nn.Conv(self.dim, [int(3)])(x)
            x = ResnetBlock()(x, time_emb)
            x = nn.ConvTranspose(self.dim, [int(3)], [int(2)])(x)

        # Final ResNet block and output convolutional layer
        x = ResnetBlock()(x, time_emb)
        x = nn.Conv(channels, [int(3)], padding="SAME")(x)
        # x = x[:, 0:original_stamp_size, 0:original_stamp_size, :]

        return x


def create_UNet_model():
    return UNet()
