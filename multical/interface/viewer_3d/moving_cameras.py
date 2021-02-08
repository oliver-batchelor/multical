
import numpy as np
from .marker import AxisSet, SceneMeshes, CameraSet, BoardSet
from multical import tables




class MovingCameras(object):
  def __init__(self, viewer, calib, board_colors):
    self.viewer = viewer
    self.view_poses = tables.inverse(tables.expand_views(calib.pose_estimates))
    self.meshes = SceneMeshes(calib)

    self.board_set = BoardSet(self.viewer, calib.pose_estimates.board, self.meshes.board, board_colors)
    self.camera_sets = [CameraSet(self.viewer, poses, self.meshes.camera)
        for poses in self.view_poses._sequence(1)]


    self.axis_set = AxisSet(self.viewer, self.meshes.axis, self.view_poses._index[:, 0])
    self.show(False)

  def update_calibration(self, calib):
      self.meshes.update(calib)
      self.view_poses = tables.inverse(tables.expand_views(calib.pose_estimates))

      for camera_set, pose in zip(self.camera_sets, self.view_poses._sequence(1)):
       camera_set.update_poses(pose)

      self.board_set.update_poses(calib.pose_estimates.board)

  def show(self, shown):
    for marker in (self.camera_sets + [self.board_set, self.axis_set]):
      marker.show(shown)

  def update(self, state):
    self.meshes.set_camera_scale(state.scale)
    for i, camera_set in enumerate(self.camera_sets):
      camera_set.update(active=(i == state.frame), highlight=state.camera)

    self.axis_set.update_poses(self.view_poses._index[:, state.frame])
    self.viewer.update()

  def enable(self, state):
    self.update(state)
    self.show(True)

  def disable(self):
    self.show(False)



