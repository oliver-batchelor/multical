
import numpy as np
from multical.interface.marker import board_object

import multical.tables as tables


class CameraView(object):
  def __init__(self, viewer, calib):
    self.viewer = viewer
    self.saved_camera=None

    self.view_poses = tables.multiply_masked(
      calib.pose_estimates.camera, calib.pose_estimates.rig)
    self.cameras = calib.cameras
    self.board = board_object(self.viewer, calib.board)
    self.show(False)


  def show(self, is_shown):
    self.board.SetVisibility(is_shown)

  def update(self, state):

    pose = self.view_poses._index[state.camera, state.frame]
    self.show(pose.valid_poses)

    viewport = self.viewer.camera_viewport(
        self.cameras[state.camera], np.linalg.inv(pose.poses))

    self.viewer.set_viewport(viewport)
    self.viewer.update()

  def enable(self, state):
    self.saved_camera = self.viewer.current_viewport()

    self.viewer.enable(False)
    self.update(state)

  def disable(self):
    self.show(False)
    self.viewer.enable(True)
    if self.saved_camera is not None:
      self.viewer.set_viewport(self.saved_camera)



