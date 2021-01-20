from calibrate import calibrate_cameras
import os
from multical.optimization.calibration import Calibration
from structs.struct import struct
from . import tables, image

class Workspace:
  def __init__(self):

    self.calibrations = {}
    self.detections = None
    self.boards = None

    self.image_names = None
    self.camera_names = None

    self.image_sizes = None
    self.images = None

    self.pose_detections = None


  def find_images(self, image_path, camera_dirs=None):
    camera_names, image_names, filenames = image.find.find_images(image_path, camera_dirs)
    print("Found camera directories {} with {} matching images".format(str(camera_names), len(image_names)))

    self.camera_names = camera_names
    self.image_names = image_names
    self.filenames = filenames

    self.image_path = image_path


  def load_detect(self, boards, j=len(os.sched_getaffinity(0))):
    assert self.filenames is not None 
    self.boards = boards

    print("Detecting patterns..")
    loaded = image.detect.detect_images(self.boards, self.filenames, j=j, prefix=self.image_path)   

    self.detected_points = loaded.points
    self.point_table = tables.make_point_table(loaded.points, self.boards)

    self.images = loaded.images
    self.image_size = loaded.image_size


  def calibrate_single(self, camera_model, fix_aspect=False):
    assert self.detected_points is not None

    print("Calibrating single cameras..")
    self.cameras, errs = calibrate_cameras(self.boards, self.detected_points, 
      self.image_size, model=camera_model, fix_aspect=fix_aspect)
    
    for name, camera, err in zip(self.camera_names, self.cameras, errs):
      print(f"Calibrated {name}, with RMS={err:.2f}")
      print(camera)
      print("---------------")


  def initialise_poses(self):
    assert self.cameras is not None

    self.pose_detections = tables.make_pose_table(self.point_table, self.boards, self.cameras)
    self.pose_estimates = tables.initialise_poses(self.pose_detections)
    calib = Calibration(self.cameras, self.boards, self.point_table, self.pose_estimates, self.pose_detections)
    self.calibrations['initialisation'] = calib

    return calib


  def calibrate(self, name, optimize=struct(intrinsics=False, board=False), **opt_args):
    calib = self.calib.enable_intrinsics(optimize.intrinsics).enable_board(optimize.board)
    calib.adjust_outliers(**opt_args)

    self.calibrations[name] = calib
    return calib

  def has_calibrations(self):
    return len(self.calibrations) > 0

  def get_calibrations(self):
    return self.calibrations

  def get_camera_sets(self):
      if self.has_calibrations():
        return {k:calib.cameras for k, calib in self.calibrations.items()}

      if self.cameras is not None:
        return dict(initialisation = self.cameras)

        
