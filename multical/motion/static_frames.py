from cached_property import cached_property
from multical.motion.motion_model import MotionModel
from multical.optimization.pose_set import PoseSet
import numpy as np
from structs.numpy import Table, shape
from structs.struct import struct, subset
from multical import tables


def project_cameras(cameras, local_points):
  image_points = [camera.project(p) for camera, p in 
      zip(cameras, local_points.points)]

  return Table.create(points=np.stack(image_points), valid=local_points.valid)

def project_points(pose_table, cameras, camera_poses, world_points):
  pose_estimates = struct(camera = camera_poses,  times=pose_table)
  pose_table = tables.expand_views(pose_estimates)

  points = tables.transform_points( 
    tables.expand_dims(pose_table, (2, 3)), 
    tables.expand_dims(world_points, (0, 1))
  )

  return project_cameras(cameras, points)



class StaticFrames(PoseSet, MotionModel):
  def __init__(self, pose_table, names):
    super(StaticFrames, self).__init__(pose_table, names)

  def project(self, cameras, camera_poses, world_points, estimates=None):
    return project_points(self.pose_table, cameras, camera_poses, world_points)
  
  @staticmethod
  def init(pose_table, names=None):
    return StaticFrames(pose_table, names)

  @cached_property
  def frame_poses(self):
    return self.pose_table
