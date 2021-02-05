
import numpy as np
from .marker import View, board_mesh, view_marker
from structs.numpy import shape
from multical import tables

def camera_markers(viewer, camera_poses, cameras, scale=1.0):
  def add_view(camera_pose, camera):
    pose = np.linalg.inv(camera_pose.poses) if camera_pose.valid else None
    return View(viewer, view_marker(camera), pose, scale)

  return [add_view(camera_pose, camera)
    for camera_pose, camera in zip(camera_poses._sequence(), cameras)]
 
def set_poses(camera_poses, views):
  for camera_pose, view in zip(camera_poses._sequence(), views):
    pose = np.linalg.inv(camera_pose.poses) if camera_pose.valid else None
    view.set_pose(pose)


def board_objects(viewer, board, pose_estimates, color):
  mesh = board_mesh(board)

  def add_board(pose):
    if pose.valid:
      return viewer.add_mesh(mesh.copy(), style="wireframe", ambient=0.5,
        transform=pose.poses, color=color, show_edges=True)

  return [add_board(pose) for pose in pose_estimates._sequence()]
   

class MovingBoard(object):
  def __init__(self, viewer, board_colors, scale=0.05):
    self.viewer = viewer

    self.scale = scale
    self.board_colors = board_colors

    self.views = []
    self.boards = []
    self.calib = None

    self.show(False)


  def set_calibration(self, calib):
    
    if self.calib is None:
      self.views = camera_markers(self.viewer, calib.pose_estimates.camera, 
        calib.cameras, scale=self.scale)

      board_poses = tables.expand_boards(calib.pose_estimates)
      self.boards = [board_objects(self.viewer, board, board_pose, color)
        for board, color, board_pose in zip(calib.boards, self.board_colors, board_poses._sequence(1))]
    else:



  def show(self, shown):
    for view in self.views:
      if view is not None: view.show(shown)

    for board_frames in self.boards:
      for board in board_frames:
        if board is not None: board.SetVisibility(shown)

  def update(self, state):

    for i, view in enumerate(self.views):
        color = (1, 1, 0) if i == state.camera else (0.5, 1, 0.0)
        view.set_color(color)
        view.set_scale(state.scale)


    for board_color, board_frames in zip(self.board_colors, self.boards):
      for i, board in enumerate(board_frames):
        color = board_color if i == state.frame else (0.5, 0.5, 0.5)
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