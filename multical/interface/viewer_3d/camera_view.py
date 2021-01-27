
import numpy as np
from .marker import board_object, image_projection

import multical.tables as tables
import pyvista as pv

import cv2

def add_image_projection(viewer, camera, pose, image):
  mesh = image_projection(camera)

  if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  texture = pv.numpy_to_texture(image)
  return viewer.add_mesh(mesh, texture=texture, lighting=False, transform=np.linalg.inv(pose.poses))

class CameraView(object):
  def __init__(self, viewer, calib, undistorted_images):
    self.viewer = viewer
    self.saved_camera=None

    self.view_poses = tables.expand_poses(calib.pose_estimates)
      
    self.cameras = calib.cameras
    self.board = board_object(self.viewer, calib.board)

    self.projections = [[add_image_projection(viewer, camera, pose, image) for image, pose in zip(cam_images, frame_poses._sequence())]
      for camera, frame_poses, cam_images in zip(self.cameras, self.view_poses._sequence(), undistorted_images)]

    self.hide()


  def hide(self):
    self.board.SetVisibility(False)
    for frames in self.projections:
      for proj in frames:
        proj.SetVisibility(False)

  def update(self, state):
    self.hide()

    pose = self.view_poses._index[state.camera, state.frame]
    self.board.SetVisibility(pose.valid)

    proj = self.projections[state.camera][state.frame]
    proj.SetVisibility(True)

    viewport = self.viewer.camera_viewport(
        self.cameras[state.camera], np.linalg.inv(pose.poses))

    self.viewer.set_viewport(viewport)
    self.viewer.update()

  def enable(self, state):
    self.saved_camera = self.viewer.current_viewport()

    self.viewer.enable(False)
    self.update(state)

  def disable(self):
    self.hide()
    self.viewer.enable(True)
    if self.saved_camera is not None:
      self.viewer.set_viewport(self.saved_camera)



