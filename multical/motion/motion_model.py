

class MotionModel():
  def project(self, cameras, camera_poses, board_poses, board_points, estimates=None):
    raise NotImplemented()

  @property
  def frame_poses(self):
    raise NotImplemented()