
import numpy as np
from structs.struct import struct
from .marker import SceneMeshes, CameraSet, BoardSet
from structs.numpy import shape
from multical import tables

import pyvista as pv

  
class MovingBoard(object):
  def __init__(self, viewer, calib, board_colors, scale=1):
    self.viewer = viewer
    self.scale = scale
    self.board_colors = board_colors

    self.meshes = SceneMeshes(calib)
    self.camera_set = CameraSet(self.viewer, calib.pose_estimates.camera, 
      self.meshes.camera, scale=self.scale)

    board_poses = tables.expand_boards(calib.pose_estimates)
    self.board_sets = [
      BoardSet(self.viewer, poses, self.meshes.board, board_colors)
        for poses in board_poses._sequence()]


  def update_calibration(self, calib):
    board_poses = tables.expand_boards(calib.pose_estimates)

    self.meshes.update(calib)
    self.camera_set.update_poses(calib.pose_estimates.camera)

    board_poses = tables.expand_boards(calib.pose_estimates)
    for board_set, poses in zip(self.board_sets, board_poses.sequence()):
      board_set.update_poses(poses)


  def show(self, shown):

    self.camera_set.show(shown)
    for board_set in self.board_sets:
      board_set.show(shown)

  def update(self, state):
    self.camera_set.update(highlight=state.camera, scale=state.scale)
    for i, board_set in enumerate(self.board_sets):
      board_set.update(active = i == state.frame)

    self.viewer.update()

    
  def enable(self, state):
    self.show(True)
    self.update(state)

  def disable(self):
    self.show(False)