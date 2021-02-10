import numpy as np
from structs.numpy import Table
from structs.struct import struct
from multical.optimization.parameters import Parameters
from multical import tables


class Static(Parameters):
  def __init__(self, time_poses):
    self.poses = time_poses


  def project(self, cameras, camera_poses, board_poses, board_points, estimates=None):
    pose_estimates = struct(camera = camera_poses, board=board_poses, times=self.time_poses)

    pose_table = tables.expand_views(pose_estimates)
    transformed = tables.transform_points(board_points, pose_table)

    image_points = [camera.project(p) for camera, p in 
      zip(self.cameras, transformed.points)]

    return Table.create(points=np.stack(image_points), valid=transformed.valid)

    
  def params(self):
    pass

  def with_params(self, params):
    pass


  def sparsity(self, index_table):
    pass