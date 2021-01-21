from structs.numpy import shape
from calibrate import calibrate_cameras
import os
from multical.optimization.calibration import Calibration
from structs.struct import split_dict, struct
from . import tables, image

from logging import getLogger, info, warning, debug




class Workspace:
  def __init__(self):

    self.calibrations = {}
    self.detections = None
    self.boards = None

    self.names = struct()

    self.image_sizes = None
    self.images = None

    self.detected_poses = None


  def find_images(self, image_path, camera_dirs=None):
    camera_names, image_names, filenames = image.find.find_images(image_path, camera_dirs)
    info("Found camera directories {} with {} matching images".format(str(camera_names), len(image_names)))

    self.names = self.names._extend(camera = camera_names, image = image_names)
    self.filenames = filenames

    self.image_path = image_path


  def load_detect(self, boards, j=len(os.sched_getaffinity(0))):
    assert self.filenames is not None 
    board_names, self.boards = split_dict(boards)
    self.names = self.names._extend(board = board_names)
    
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
    
    for name, camera, err in zip(self.names.camera, self.cameras, errs):
      print(f"Calibrated {name}, with RMS={err:.2f}")
      print(camera)
      print("---------------")


  def initialise_poses(self):
    assert self.cameras is not None

    self.detected_poses = tables.make_pose_table(self.point_table, self.boards, self.cameras)
    
    pose_initialisation = tables.initialise_poses(self.detected_poses)
    calib = Calibration(self.cameras, self.boards, self.point_table, pose_initialisation)
    self.calibrations['initialisation'] = calib

    return calib


  def calibrate(self, name, optimize=struct(intrinsics=False, board=False), **opt_args):
    calib = self.calib.enable_intrinsics(optimize.intrinsics).enable_board(optimize.board)
    calib.adjust_outliers(**opt_args)

    self.calibrations[name] = calib
    return calib

  @property
  def sizes(self):
    return self.names._map(len)

  def has_calibrations(self):
    return len(self.calibrations) > 0

  def get_calibrations(self):
    return self.calibrations

  def get_camera_sets(self):
      if self.has_calibrations():
        return {k:calib.cameras for k, calib in self.calibrations.items()}

      if self.cameras is not None:
        return dict(initialisation = self.cameras)

        
