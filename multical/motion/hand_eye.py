from multical.motion.static_frames import project_points
import numpy as np
from structs.struct import struct, subset
from multical import tables
from multical.io.export_calib import export_poses
from multical.optimization.parameters import IndexMapper, Parameters
from cached_property import cached_property
from multical.motion.motion_model import MotionModel

from multical.transform import rtvec as transform_vec



class HandEye(Parameters, MotionModel):
  """
  Optimises the world to camera rig transform (frames) constrained to a 
  'robot world' hand-eye calibration form where world_wrt_base and gripper_wrt_camera are 
  optimised and a set of base_gripper transforms are provided (and constant).

  world_wrt_camera = gripper_wrt_camera @ base_wrt_gripper @ world_wrt_base
  """
  def __init__(self, base_wrt_gripper, world_wrt_base, gripper_wrt_camera,  names=None):
    self.base_wrt_gripper = base_wrt_gripper
    n = base_wrt_gripper._shape[0]

    self.names = names or [str(i) for i in range(n)]

    self.world_wrt_base = world_wrt_base
    self.gripper_wrt_camera = gripper_wrt_camera


  @property
  def size(self):
    return self.pose_table._shape[0]

  def project(self, cameras, camera_poses, world_points, estimates=None):
    return project_points(self.pose_table, cameras, camera_poses, world_points)

  @cached_property
  def valid(self):
    return self.base_wrt_gripper.valid

  @cached_property
  def pose_table(self):
    base_wrt_camera = tables.multiply(self.gripper_wrt_camera, self.base_wrt_gripper)
    return tables.multiply(base_wrt_camera, self.world_wrt_base)

  @property
  def frame_poses(self):
    return self.pose_table

  def pre_transform(self, t):   
    return self.copy(gripper_wrt_camera = t @ self.gripper_wrt_camera)

  def post_transform(self, t):   
    return self.copy(world_wrt_base = self.world_wrt_base @ t)


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

  @cached_property
  def params(self):
    return struct(
      world_wrt_base = transform_vec.from_matrix(self.world_wrt_base),
      gripper_wrt_camera = transform_vec.from_matrix(self.gripper_wrt_camera)
    )

  def with_params(self, params):
    return self.copy(
      world_wrt_base = transform_vec.to_matrix(params.world_wrt_base),
      gripper_wrt_camera = transform_vec.to_matrix(params.gripper_wrt_camera)
    )


  def sparsity(self, index_mapper : IndexMapper, axis : int):
    return index_mapper.all_points(transform_vec.size * 2)

  def export(self):
    return struct(
      base_wrt_gripper = export_poses(self.base_wrt_gripper, self.names),
      world_wrt_base = self.world_wrt_base.tolist(),
      gripper_wrt_camera = self.gripper_wrt_camera.tolist()
    )

  def __getstate__(self):
    attrs = ['base_wrt_gripper', 'gripper_wrt_camera', 'world_wrt_base', 'names']
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy object and change some attribute (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return self.__class__(**d)