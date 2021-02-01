
import numpy as np
from .marker import board_object, View, view_marker
from multical import tables

from numba import jit

@jit(forceobj=True)
def view_markers(viewer, pose_estimates, cameras, scale=1.0):
  view_poses = tables.inverse(tables.expand_views(pose_estimates))

  def add_view(view_pose, camera):
    if view_pose.valid:
      return View(viewer, camera, view_pose.poses, scale)

  marker_meshes = [view_marker(camera) for camera in cameras]
         
  return [[add_view(view_pose, marker.copy())
    for view_pose, marker in zip(camera_poses._sequence(), marker_meshes)]
        for camera_poses in view_poses._sequence(1)]


class MovingCameras(object):
  @jit(forceobj=True)
  def __init__(self, viewer, calib, board_colors):
    self.viewer = viewer

    board_poses = calib.pose_estimates.board.poses

    self.views = view_markers(self.viewer, calib.pose_estimates, calib.cameras)
    self.boards = [board_object(self.viewer, board, color, transform=t) 
      for board, color, t in zip(calib.boards, board_colors, board_poses)]

  def show(self, is_shown):
    for board in self.boards:
      board.SetVisibility(is_shown)
  
    for view in self.valid_views:
      view.show(is_shown)

  @property
  def valid_views(self):
    return [view for frame_views in self.views
      for view in frame_views
        if view is not None]

  def update(self, state):
    for i, frame_views in enumerate(self.views):
      for j, view in enumerate(frame_views): 
          color = (0.5, 0.5, 0.5)
          if i == state.frame: 
            color = (1, 1, 0) if j == state.camera else (0.5, 1, 0)

          if view is not None:
            view.set_color(color)
            view.set_scale(state.scale)

    self.viewer.update()

  def enable(self, state):
    self.show(True)
    self.update(state)

  def disable(self):
    self.show(False)



