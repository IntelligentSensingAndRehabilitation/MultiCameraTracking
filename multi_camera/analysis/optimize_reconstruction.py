""" 
This file uses implicit functions to reconstruct the 3D joint locations
from the 2D joint locations. This is an alternative to triangulation that
allows adding additional constraints such as constant limb lengths and also
allows more flexibly reweighting the different reprojection losses compared
to pure triangulation. Using an implict function is an alternative to directly
optimizing the trajectory sequences subject to similar constraints. The hope
is that this will be more robust to noise and outliers, and ultimately can
be used in a more flexible inference process that meta-learns over multiple
trajectories.

It uses Jax and flax to define the implicit function and optax to optimize.

Motivate by this clean implication of a NeRF:
https://github.com/google-research/google-research/blob/master/trainable_grids/Voxel_based_Radiance_Fields.ipynb

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0

"""

from typing import Callable, Sequence, Tuple

Initializer = Callable

from functools import partial
import dataclasses
import time

import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.core.frozen_dict import FrozenDict

import optax


class KeypointTrajectory(nn.Module):
    """ A simple module that stores a trajectory of 3D keypoints and also finer time scale interpolation. """

    size: int
    joints: int = 17
    spatial_dims: int = 3
    grid_init: Initializer = nn.initializers.lecun_normal()

    @staticmethod
    def interpolate(feature_grid, x):
        """Interpolate vectors on a regular grid at samples 0, 1, ... n-1."""
        xd, whole = jnp.modf(x)  # 3.6 -> xd = 0.6
        x0 = whole.astype(int)  # x0 = 3
        x1 = x0 + 1  # z1 = 4

        def f(grid):
            return grid[x0] * (1.0 - xd) + grid[x1] * xd

        return jax.vmap(f, -1, -1)(feature_grid)

    @nn.compact
    def __call__(self, x):
        grid = self.param('grid', self.grid_init, (self.size + 1, self.joints * self.spatial_dims))
        return self.interpolate(grid, x).reshape(-1, self.joints, self.spatial_dims) * 1000.0 # represent in m

def positional_encoding(inputs, positional_encoding_dims=3):
    batch_size, _ = inputs.shape
    inputs_freq = jax.vmap(
        lambda x: inputs * 2.0 ** x
    )(jnp.arange(positional_encoding_dims))
    #print(inputs_freq)
    periodic_fns = jnp.stack([jnp.sin(inputs_freq), jnp.cos(inputs_freq)])
    periodic_fns = periodic_fns.swapaxes(0, 2).reshape([batch_size, -1])
    periodic_fns = jnp.concatenate([inputs, periodic_fns], axis=-1)
    return periodic_fns


class ImplicitTrajectory(nn.Module):
    """ This module defines the implicit function that maps from time to the 3D joint locations. """

    size: int
    features: Sequence[int]
    joints: int = 17
    spatial_dims: int = 3
    dense_init: Initializer = nn.initializers.lecun_normal()
    concatenate_layers = 4
    #dense_init: Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, input_points):

        # works better when using time scale (0,1)
        input_points = input_points / self.size
        input_points = positional_encoding(input_points, self.features[0] // 2)

        x = input_points
        
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=self.dense_init)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            if i < self.concatenate_layers:
                x = jnp.concatenate([x, input_points], axis=-1)
        x = nn.Dense(self.joints * self.spatial_dims)(x)
        x = x.reshape(-1, self.joints, self.spatial_dims)
        return x * 1000.0

        #grid = self.param('grid', self.dense_init, (2, self.joints * self.spatial_dims))
        #x = input_points * grid[0, None, :] + grid[1, None, :]
        #x = x.reshape(-1, self.joints, self.spatial_dims)
        #return x * 1000.0


def huber(x, delta=5.0, max=30, max_slope=0.1):
    """Huber loss."""
    x = jnp.where(jnp.abs(x) < delta, 0.5 * x ** 2, delta * (jnp.abs(x) - 0.5 * delta))
    #x = jnp.where(x > max, (x - max) * max_slope + max, x)
    return x


def reprojection_loss(camera_params, points2d, points3d):
    """Compute reprojection loss between 3D keypoints and 2D keypoints."""

    from .camera import reprojection_error

    conf = points2d[..., -1]
    conf = conf * (conf > 0.5) # only use points with high confidence
    loss = reprojection_error(camera_params, points2d[..., :-1], points3d)
    delta = huber(jnp.linalg.norm(loss, axis=-1))
    #delta = jnp.linalg.norm(loss, axis=-1)
    return jnp.nanmean(delta * conf)


def smoothness_loss(points3d):
    """Compute reprojection loss between 3D keypoints and 2D keypoints."""

    delta = jnp.diff(points3d, axis=0)
    delta = jnp.linalg.norm(delta, axis=-1)
    return jnp.nanmean(delta)

def relative_smoothness_loss(points3d, reference_joint):
    """Compute reprojection loss between 3D keypoints and 2D keypoints."""

    points3d = points3d - points3d[:, reference_joint, None, :]
    delta = jnp.diff(points3d, axis=0)
    delta = jnp.linalg.norm(delta, axis=-1)
    return jnp.nanmean(delta)

def skeleton_loss(points3d, skeleton_pairs):
    """Compute change in limb lengths"""

    limb_lengths = points3d[:, skeleton_pairs[:, 0]] - points3d[:, skeleton_pairs[:, 1]]
    delta_limb_lengths = jnp.diff(limb_lengths, axis=0)
    delta_limb_lengths = jnp.linalg.norm(delta_limb_lengths, axis=-1)
    return jnp.nanmean(delta_limb_lengths)


def build_explicit(keypoints2d):
    n_steps = keypoints2d.shape[1]
    n_joints = keypoints2d.shape[2]
    print(f'n_steps: {n_steps}, n_joints: {n_joints}')
    model = KeypointTrajectory(size=n_steps, joints=n_joints, spatial_dims=3)
    return model

def build_implicit(keypoints2d):
    n_steps = keypoints2d.shape[1]
    n_joints = keypoints2d.shape[2]
    model = ImplicitTrajectory(size=n_steps, features=[64, 128, 256, 512, 1024], joints=n_joints, spatial_dims=3)
    return model


def optimize_trajectory(keypoints2d, camera_params, method='implicit', skeleton=None, learning_rate=None, 
                       return_model=False, seed=0, skeleton_weight=0.0, delta_weight=0.0, max_iters=2000): 

    if method == 'implicit':
        model = build_implicit(keypoints2d)
    elif method == 'explicit':
        model = build_explicit(keypoints2d)

    if learning_rate is None:
        if method == 'implicit':
            learning_rate = optax.linear_schedule(init_value=1e-3, end_value=1e-5, transition_steps=max_iters)
        else:
            learning_rate = 1e-1

    x = np.arange(keypoints2d.shape[1])[:, None]

    variables = model.init(jax.random.PRNGKey(seed), x)

    @jax.jit
    def loss_fn(variables, delta_weight=delta_weight, skeleton_weight=skeleton_weight):
        pred = model.apply(variables, x)
        l_repro = reprojection_loss(camera_params, keypoints2d, pred)
        l_delta = smoothness_loss(pred) #+ relative_smoothness_loss(pred, 19) # pelvis
        if skeleton is None:
            l_skeleton = 0.0
        else:
            l_skeleton = skeleton_loss(pred, skeleton)
        return l_repro + l_delta * delta_weight + l_skeleton * skeleton_weight
        
    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(variables)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    for i in range(max_iters):    
        loss_val, grads = loss_grad_fn(variables)
        updates, opt_state = tx.update(grads, opt_state)
        variables = optax.apply_updates(variables, updates)
        if i % np.ceil(max_iters / 20) == 0:
            print('Loss step {}: '.format(i), loss_val)

    pred = model.apply(variables, x)

    if return_model:

        losses = {'reprojection_loss': reprojection_loss(camera_params, keypoints2d, pred), 
                  'smoothness_loss': smoothness_loss(pred)}

        if skeleton is not None:
            losses['skeleton_loss'] = skeleton_loss(pred, skeleton)

        return pred, {'model': model, 'variables': variables, **losses}
    
    return pred
