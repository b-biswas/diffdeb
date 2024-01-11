import logging
import os

from jax.lib import xla_bridge

from diffdeb.config import get_config_diffusion
from diffdeb.dataset import batched_CATSIMDataset
from diffdeb.train_UNet import train_and_evaluate_UNet

logging.basicConfig(level=logging.INFO)

# Check if JAX is using GPU.
logging.info(xla_bridge.get_backend().platform)
config = get_config_diffusion()

# Load the dataset.
ds_isolated_train, ds_isolated_val = batched_CATSIMDataset(
    tf_dataset_dir=os.path.join(
        "/sps/lsst/users/bbiswas/simulations/LSST/",
        "isolated_tfDataset",
    ),
    linear_norm_coeff=config.linear_norm_coeff,
    batch_size=config.batch_size,
    x_col_name="blended_gal_stamps",
    y_col_name="isolated_gal_stamps",
)

# Launch UNet training.
train_and_evaluate_UNet(
    train_tfds=ds_isolated_train,
    val_tfds=ds_isolated_val,
    config=config,
)
