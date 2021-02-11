from multical.io.export import export_poses
from cached_property import cached_property
import numpy as np
from structs.numpy import Table
from structs.struct import struct, subset
from multical.optimization.parameters import IndexMapper, Parameters
from multical import tables

from multical.transform import rtvec


class Static(Parameters):
  def __init__(self, frame_poses):
    self.frame_poses = frame_poses

  def project(self, cameras, camera_poses, board_poses, board_points, estimates=None):
    pose_estimates = struct(camera = camera_poses, board=board_poses, times=self.frame_poses)

    pose_table = tables.expand_poses(pose_estimates)
    transformed = tables.transform_points(pose_table, board_points)

    image_points = [camera.project(p) for camera, p in 
      zip(cameras, transformed.points)]

    return Table.create(points=np.stack(image_points), valid=transformed.valid)

  @cached_property
  def valid_frames(self):
    return self.frame_poses.valid

  @cached_property
  def params(self):
    return rtvec.from_matrix(self.frame_poses.poses).ravel()

  def with_params(self, params):
    m = rtvec.to_matrix(params.reshape(-1, 6))
    return self.copy(frame_poses = self.frame_poses._update(poses=m))

  def sparsity(self, index_mapper : IndexMapper):
    return index_mapper.pose_mapping(self.frame_poses, axis=1)

  def export(self):
    return struct(frames = export_poses(self.frame_poses))

  def __getstate__(self):
    attrs = ['frame_poses']
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy calibration environment and change some parameters (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return Static(**d)