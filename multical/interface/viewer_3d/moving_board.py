
import numpy as np
from structs.struct import struct
from .marker import AxisSet, SceneMeshes, CameraSet, BoardSet
from structs.numpy import shape
from multical import tables

import pyvista as pv

  
class MovingBoard(object):
  def __init__(self, viewer, calib, board_colors):
    self.viewer = viewer
    self.board_colors = board_colors
    self.meshes = SceneMeshes(calib)

    camera_poses = tables.inverse(calib.pose_estimates.camera)
    self.camera_set = CameraSet(self.viewer, camera_poses, self.meshes.camera)

    board_poses = tables.expand_boards(calib.pose_estimates)
    self.board_sets = [
      BoardSet(self.viewer, poses, self.meshes.board, board_colors)
        for poses in board_poses._sequence()]

    self.axis_set = AxisSet(self.viewer, self.meshes.axis, camera_poses)
    self.show(False)

  def update_calibration(self, calib):
    self.meshes.update(calib)
    
    board_poses = tables.expand_boards(calib.pose_estimates)
    camera_poses = tables.inverse(calib.pose_estimates.camera)
    self.camera_set.update_poses(camera_poses)
    self.axis_set.update_poses(camera_poses)

    board_poses = tables.expand_boards(calib.pose_estimates)
    for board_set, poses in zip(self.board_sets, board_poses._sequence()):
      board_set.update_poses(poses)

  def show(self, shown):
    for marker in self.board_sets + [self.camera_set, self.axis_set]:
      marker.show(shown)

  def update(self, state):
    self.meshes.set_camera_scale(state.scale)
    self.camera_set.update(highlight=state.camera)
    for i, board_set in enumerate(self.board_sets):
      board_set.update(active = i == state.frame)

    self.viewer.update()

    
  def enable(self, state):
    self.show(True)
    self.update(state)

  def disable(self):
    self.show(False)