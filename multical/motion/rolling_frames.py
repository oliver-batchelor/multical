from multical.motion.static_frames import project_cameras
from cached_property import cached_property
import numpy as np
from structs.struct import struct, subset
from .motion_model import MotionModel

from multical.optimization.parameters import IndexMapper, Parameters
from multical import tables
from structs.numpy import Table, shape

from multical.transform import rtvec, matrix
from multical.transform.interpolate import interpolate_poses, lerp


def rolling_times(cameras, point_table):
  image_heights = np.array([camera.image_size[1] for camera in cameras])    
  times = point_table.points[..., 1] / np.expand_dims(image_heights, (1,2,3))  

  return times

def transformed_linear(self, camera_poses, world_points, times):
  """ A linear approximation where the points are transformed at the start.
      and end of the frame. Points are then computed by linear interpolation in between
 """

  def transform(time_poses):
    pose_table = tables.expand_views(
        struct(camera=camera_poses, times=time_poses))

    return tables.transform_points( 
      tables.expand_dims(pose_table, (2, 3)), 
      tables.expand_dims(world_points, (0, 1))
    )

  start_frame = transform(self.start_table)
  end_frame = transform(self.end_table)

  return struct(
    points = lerp(start_frame.points, end_frame.points, times),
    valid = start_frame.valid & end_frame.valid
  )


def transformed_interpolate(self, camera_poses, world_points, times):
  """ Compute in-between transforms using interpolated poses 
  (quanternion slerp and lerp)
  """
  start_frame = np.expand_dims(self.pose_start, (0, 2, 3))
  end_frame = np.expand_dims(self.pose_end, (0, 2, 3))
  
  frame_poses = interpolate_poses(start_frame, end_frame, times)
  view_poses = np.expand_dims(camera_poses.poses, (1, 2, 3)) @ frame_poses  

  valid = (np.expand_dims(camera_poses.valid, [1, 2, 3]) & 
    np.expand_dims(self.valid, [0, 2, 3]) &
    np.expand_dims(world_points.valid, [0, 1]))

  return struct(
    points = matrix.transform_homog(t = view_poses, 
      points=np.expand_dims(world_points.points, (0, 1))),

    valid = valid
  )


class RollingFrames(MotionModel, Parameters):

  def __init__(self, pose_start, pose_end, valid, names, max_iterations=4):
    self.pose_start = pose_start
    self.pose_end = pose_end
    self.valid = valid
    self.names = names
    self.max_iterations = max_iterations

  @property
  def size(self):
    return self.poses.shape[0]

  @cached_property
  def valid(self):
    return self.pose_table.valid

  @cached_property
  def start_table(self):
     return Table.create(poses=self.pose_start, valid=self.valid)
  
  @cached_property
  def end_table(self):
    return Table.create(poses=self.pose_end, valid=self.valid)
  
  @cached_property
  def frame_poses(self):
    return self.start_table

  def pre_transform(self, t):   
    return self.copy(
        pose_start = t @ self.pose_start, 
        pose_end = t @ self.pose_end
      )        

  def post_transform(self, t):   
    return self.copy(
        pose_start = self.pose_start @ t, 
        pose_end = self.pose_end @ t
      )

  @staticmethod
  def init(pose_table, names=None, max_iterations=4):
    size = pose_table.valid.size
    names = names or [str(i) for i in range(size)]

    return RollingFrames(pose_table.poses, pose_table.poses, 
      pose_table.valid, names, max_iterations=max_iterations)
 
  def _project(self, cameras, camera_poses, world_points, estimates=None):
    num_frames = self.pose_start.shape[0]
    num_cameras = camera_poses.valid.shape[0]

    times = rolling_times(cameras, estimates) if estimates is not None\
      else np.full((num_cameras, num_frames, *world_points.valid.shape), 0.5)
    
    transformed = transformed_linear(self, camera_poses, world_points, times)
    return project_cameras(cameras, transformed)

  def project(self, cameras, camera_poses, world_points, estimates=None):
    points = self._project(cameras, camera_poses, world_points, estimates)

    if estimates is None:
      for i in range(0, self.max_iterations):
        points = self._project(cameras, camera_poses, world_points, points)
    

    return points

  @cached_property
  def params(self):
    return [
      rtvec.from_matrix(self.pose_start).ravel(),
      rtvec.from_matrix(self.pose_end).ravel()
    ]
      
  def with_params(self, params):
    start, end = [rtvec.to_matrix(m.reshape(-1, 6)) for m in params]
    return self.copy(pose_start=start, pose_end=end)

  def sparsity(self, index_mapper : IndexMapper, axis : int):
    start, end = [index_mapper.pose_mapping(t, axis=axis, param_size=6) 
      for t in [self.start_table, self.end_table]]

    return start + end
    

  def export(self):
    return {i:struct(start=start.tolist(), end=end.tolist()) 
      for i, start, end, valid in zip(self.names, self.pose_start, self.pose_end, self.valid) 
        if valid}

  def __getstate__(self):
    attrs = ['pose_start', 'pose_end', 'valid', 'names', 'max_iterations']
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy object and change some attribute (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return self.__class__(**d)
  




 
