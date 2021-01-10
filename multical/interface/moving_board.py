
import numpy as np
from multical.interface.marker import View, board_mesh


def camera_markers(viewer, camera_poses, cameras, scale=1.0):
  def add_view(camera_pose, camera):
    if camera_pose.valid_poses:
        return View(viewer, camera, np.linalg.inv(camera_pose.poses), scale)

  return [add_view(camera_pose, camera)
    for camera_pose, camera in zip(camera_poses._sequence(), cameras)]
 

def board_objects(viewer, board, pose_estimates):

  def add_board(pose):
    mesh = board_mesh(board)

    if pose.valid_poses:
      return viewer.add_mesh(mesh, style="wireframe", ambient=0.5,
        transform=pose.poses, color=(1, 0, 0), show_edges=True)

  return [add_board(pose) for pose in pose_estimates.rig._sequence()]
   

class MovingBoard(object):
  def __init__(self, viewer, calib, scale=0.05):
    self.viewer = viewer

    self.scale = scale
    self.views = camera_markers(self.viewer, calib.pose_estimates.camera, 
      calib.cameras, scale=self.scale)

    self.boards = board_objects(self.viewer, calib.board, calib.pose_estimates)
    self.board_color = (1, 0, 0)

    self.show(False)

  def show(self, shown):
    for view in self.views:
      if view is not None: view.show(shown)

    for board in self.boards:
      if board is not None: board.SetVisibility(shown)

  def update(self, state):

    for i, view in enumerate(self.views):
        color = (1, 1, 0) if i == state.camera else (0.5, 1, 0.0)
        view.set_color(color)
        view.set_scale(state.scale)

    for i, board in enumerate(self.boards):
        color = self.board_color if i == state.frame else (0.5, 0.5, 0.5)
        opacity = 1 if i == state.frame else 0.1

        if board is not None:
          p = board.GetProperty()
          p.SetColor(*color)
          p.SetOpacity(opacity)

    self.viewer.update()

    
  def enable(self, state):
    self.show(True)
    self.update(state)

  def disable(self):
    self.show(False)