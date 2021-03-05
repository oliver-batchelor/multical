from collections import OrderedDict

from natsort.natsort import natsorted
from multical.optimization.parameters import ParamList
from multical.optimization.pose_set import PoseSet
from multical.motion import StaticFrames
from os import path
from multical.io.export import export, export_cameras
from multical.image.detect import common_image_size
import numpy as np
from structs.numpy import shape
import os
from multical.optimization.calibration import Calibration
from structs.struct import map_list, split_dict, struct, subset
from . import tables, image
from .camera import calibrate_cameras

from .io.logging import MemoryHandler, info, warning, debug
from .display import make_palette
from numbers import Integral

import pickle




def num_threads():
  return len(os.sched_getaffinity(0))

def log_cameras(camera_names, cameras, errs):
  for name, camera, err in zip(camera_names, cameras, errs):
    info(f"Calibrated {name}, with RMS={err:.2f}")
    info(camera)
    info("")


class Workspace:
  def __init__(self):

    self.calibrations = OrderedDict()
    self.detections = None
    self.boards = None
    self.board_colors = None

    self.filenames = None
    self.image_path = None
    self.names = struct()

    self.image_sizes = None
    self.images = None

    self.point_table = None
    self.pose_table = None

    self.master = None

    self.log_handler = MemoryHandler()

  def try_load_detections(self, filename):
    try:
      with open(filename, "rb") as file:
        loaded = pickle.load(file)
        # Check that the detections match the metadata
        if (loaded.filenames == self.filenames and 
            loaded.boards == self.boards and
            loaded.image_sizes == self.image_sizes):

          info(f"Loaded detections from {filename}")
          return loaded.detected_points
        else:
          info(f"Config changed, not using loaded detections in {filename}")
    except (OSError, IOError, EOFError, AttributeError) as e:
      return None

  def write_detections(self, filename):
    data = struct(
      filenames = self.filenames,
      boards = self.boards,
      image_sizes = self.image_sizes,
      detected_points = self.detected_points
    )
    with open(filename, "wb") as file:
      pickle.dump(data, file)


  def find_images_matching(self, image_path, cameras=None, camera_pattern=None,  master=None, extensions=image.find.image_extensions):   
        
    camera_paths = image.find.find_cameras(image_path, cameras, camera_pattern, extensions=extensions)
    camera_names = list(camera_paths.keys())

    image_names, filenames = image.find.find_images(camera_paths, extensions=extensions)
    info("Found camera directories {} with {} matching images".format(camera_names, len(image_names)))


    self.names = self.names._extend(camera = camera_names, image = image_names)
    self.filenames = filenames
    self.image_path = image_path
    
    self.master = master or self.names.camera[0]
    assert master is None or master in self.names.camera,\
      f"master f{master} not found in cameras f{str(camera_names)}"



  def load_images(self, j=num_threads()):
    assert self.filenames is not None 

    info("Loading images..")
    self.images = image.detect.load_images(self.filenames, j=j, prefix=self.image_path)
    self.image_size = map_list(common_image_size, self.images)

    info(f"Loaded {self.sizes.image * self.sizes.camera} images")
    info({k:image_size for k, image_size in zip(self.names.camera, self.image_size)})




  def detect_boards(self, boards, cache_file=None, load_cache=True, j=num_threads()):
    assert self.boards is None
 
    board_names, self.boards = split_dict(boards)
    self.names = self.names._extend(board = board_names)
    self.board_colors = make_palette(len(boards))

    self.detected_points = self.try_load_detections(cache_file) if load_cache else None
    if self.detected_points is None:
      info("Detecting boards..")
      self.detected_points = image.detect.detect_images(self.boards, self.images, j=j)   

      if cache_file is not None:
        info(f"Writing detection cache to {cache_file}")
        self.write_detections(cache_file)


    self.point_table = tables.make_point_table(self.detected_points, self.boards)
    info("Detected point counts:")
    tables.table_info(self.point_table.valid, self.names)


  def calibrate_single(self, camera_model, fix_aspect=False, has_skew=False, max_images=None):
    assert self.detected_points is not None

    info("Calibrating single cameras..")
    self.cameras, errs = calibrate_cameras(self.boards, self.detected_points, 
      self.image_size, model=camera_model, fix_aspect=fix_aspect, has_skew=has_skew, max_images=max_images)
    
    log_cameras(self.names.camera, self.cameras, errs)


  def initialise_poses(self, motion_model=StaticFrames):
    assert self.cameras is not None
    self.pose_table = tables.make_pose_table(self.point_table, self.boards, self.cameras)
    
    info("Pose counts:")
    tables.table_info(self.pose_table.valid, self.names)

    pose_init = tables.initialise_poses(self.pose_table)

    calib = Calibration(
      ParamList(self.cameras, self.names.camera),
      ParamList(self.boards, self.names.board), 
      self.point_table, 
      PoseSet(pose_init.camera, self.names.camera), 
      PoseSet(pose_init.board, self.names.board), 
      motion_model.init(pose_init.times, self.names.image))
 
    #calib = calib.reject_outliers_quantile(0.75, 5)
    calib.report(f"Initialisation")

    self.calibrations['initialisation'] = calib
    return calib


  def calibrate(self, name, camera_poses=True, motion=True, board_poses=True, cameras=False, boards=False, **opt_args):
    calib = self.latest_calibration.enable(
        cameras=cameras, boards=boards,
        camera_poses=camera_poses, motion=motion, board_poses=board_poses
        )
    calib = calib.adjust_outliers(**opt_args)

    self.calibrations[name] = calib
    return calib

  @property
  def sizes(self):
    return self.names._map(len)

  @property
  def initialisation(self):
    return self.calibrations['initialisation']

  @property
  def latest_calibration(self):
    return list(self.calibrations.values())[-1]

  @property
  def log_entries(self):
    return self.log_handler.records

  def has_calibrations(self):
    return len(self.calibrations) > 0

  def get_calibrations(self):
    return self.calibrations

  def get_camera_sets(self):
      if self.has_calibrations():
        return {k:calib.cameras for k, calib in self.calibrations.items()}

      if self.cameras is not None:
        return dict(initialisation = self.cameras)

  def export(self, filename):
    info(f"Exporting calibration to {filename}")

    calib = self.latest_calibration
    if self.master is not None:
      calib = calib.with_master(self.master)

    export(filename, calib, self.names, master=self.master)

  def dump(self, filename):
    info(f"Dumping state and history to {filename}")
    with open(filename, "wb") as file:
      pickle.dump(self, file)

  @staticmethod
  def load(filename):
    assert path.isfile(filename), f"Workspace.load: file does not exist {filename}"
    with open(filename, "rb") as file:
      ws = pickle.load(file)
      return ws

  def __getstate__(self):
    d = subset(self.__dict__, [
       'calibrations', 'detections', 'boards', 
       'board_colors', 'filenames', 'image_path', 'names', 'image_sizes',
       'point_table', 'pose_table', 'log_handler'
    ])
    return d

  def __setstate__(self, d):
    for k, v in d.items():
      self.__dict__[k] = v

    self.images = None



