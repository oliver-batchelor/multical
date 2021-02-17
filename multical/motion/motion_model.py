

class MotionModel():
  def project(self, cameras, camera_poses, board_poses, board_points, estimates=None):
    raise NotImplementedError()

  @property
  def frame_poses(self):
    raise NotImplementedError()