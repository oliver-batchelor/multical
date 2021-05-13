from multical import tables
from structs.struct import struct, subset
from multical.optimization.parameters import Parameters
from cached_property import cached_property
from multical.io.export_calib import export_poses

from multical.transform import rtvec as transform_vec
from .parameters import IndexMapper

import numpy as np

class PoseSet(Parameters):
  def __init__(self, pose_table, names=None):
    self.pose_table = pose_table
    self.names = names or [str(i) for i in range(self.size)]

  @property
  def size(self):
    return self.poses.shape[0]

  @cached_property
  def valid(self):
    return self.pose_table.valid

  @cached_property
  def inverse(self):
    return self.copy(pose_table=tables.inverse(self.pose_table))

  def __getitem__(self, k):
    if isinstance(k, str):
      if k not in self.names:
        raise KeyError(f"pose {k} not found in {self.names}")    
      
      return self.poses[self.names.index(k)]
    else:
      return self.poses[k]

  def relative(self, src, dest):
    return self[dest] @ np.linalg.inv(self[src])

  @cached_property
  def poses(self):
    return self.pose_table.poses

  def pre_transform(self, t):   
    return self.copy(pose_table = self.pose_table._extend(poses = t @ self.poses))

  def post_transform(self, t):   
    return self.copy(pose_table = self.pose_table._extend(poses = self.poses @ t))

  @cached_property
  def params(self):
    return transform_vec.from_matrix(self.poses).ravel()

  def with_params(self, params):
    m = transform_vec.to_matrix(params.reshape(-1, transform_vec.size))
    return self.copy(pose_table = self.pose_table._update(poses=m))

  def sparsity(self, index_mapper : IndexMapper, axis : int):
    return index_mapper.pose_mapping(self.pose_table, axis=axis, param_size=transform_vec.size)

  def export(self):
    return struct(poses = export_poses(self.pose_table, self.names))

  def __getstate__(self):
    attrs = ['pose_table', 'names']
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy object and change some attribute (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return self.__class__(**d)