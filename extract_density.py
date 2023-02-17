# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation script."""

import functools
from os import path
import sys
import time

from absl import app
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import raw_utils
from internal import ref_utils
from internal import train_utils
from internal import utils
from internal import vis
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from jax.config import config as jax_config
import pickle
from tqdm import tqdm
from absl import flags
# jax_config.update('jax_disable_jit', True)

configs.define_common_flags()
flags.DEFINE_float('radius', 1., 'scene to grid radius')
flags.DEFINE_float('scene_rescale', 2./3., 'scene to world rescale')
jax.config.parse_flags_with_absl()


def main(unused_argv):
  config = configs.load_config(save_config=False)

  # dataset = datasets.load_dataset('test', config.data_dir, config)

  key = random.PRNGKey(20200823)
  _, state, density_query_pfn = train_utils.setup_model_extract_density(config, key)

  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)

  print("Model constructed")

  # points = np.zeros([1,10,1,3])
  # points[0,0,0,:] = [ 0.10009034, -2.5766413 ,  0.10569859]
  # points = jnp.array(points)
  # density= density_query_pfn(state.params, points)


  grid_size = 512

  def grid2world(points):
    roffset = flags.FLAGS.radius * (-1.)
    rscaling = 2. * flags.FLAGS.radius / grid_size
    scene_rescale = flags.FLAGS.scene_rescale
    return (roffset + points * rscaling) / scene_rescale

  X = Y = Z = jnp.arange(grid_size)
  X, Y, Z = jnp.meshgrid(X, Y, Z, indexing='ij')
  grid_pts = jnp.stack([X,Y,Z], axis=-1).reshape([-1,3])
  world_pts = grid2world(grid_pts)

  batch_size = 4096 * 4
  all_density = []
  for i in tqdm(range(0, world_pts.shape[0], batch_size)):
    density = density_query_pfn(state.params, world_pts[i:i+batch_size, :].reshape([1,-1,1,3]))
    all_density.append(density)

  all_density = jnp.concatenate(all_density, axis=2).reshape([grid_size, grid_size, grid_size, 1])

  all_density = np.array(all_density, dtype=np.float32)
  np.save(f'{config.checkpoint_dir}/multinerf_grid.npy', all_density)


if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
