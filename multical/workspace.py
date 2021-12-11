from collections import OrderedDict
from multical.threading import parmap_lists
import pathlib
from multical.board.board import Board

import numpy as np
from multical.motion import StaticFrames
from multiprocessing import cpu_count

from multical.optimization.parameters import ParamList
from multical.optimization.pose_set import PoseSet
from multical import config

from os import path
from multical.io import export_json, try_load_detections, write_detections
from multical.image.detect import common_image_size, estimate_quality

from multical.optimization.calibration import Calibration, select_threshold
from structs.struct import map_list, split_dict, struct, subset, to_dicts

from multical.transform.matrix import sample_poses
from . import tables, image
from .camera import calibrate_cameras

from structs.numpy import shape

from .io.logging import MemoryHandler, info
from .display import color_sets

import pickle
import json

def detect_boards_cached(boards, images, detections_file, cache_key, load_cache=True, j=cpu_count()):
  assert isinstance(boards, list)

  detected_points = (try_load_detections(
    detections_file, cache_key) if load_cache else None)

  if detected_points is None:
    info("Detecting boards..")
    detected_points = image.detect.detect_images(boards, images, j=j)

    info(f"Writing detection cache to {detections_file}")
    write_detections(detections_file, detected_points, cache_key)

  return detected_points

def num_valid_detections(boards, frames):
  n = 0
  for frame_detections in frames:
    for board, dets in zip(boards, frame_detections):
      if board.has_min_detections(dets): n = n + 1
  return n

def check_detections(camera_names, boards, detected_points):
  cameras = [k for k, fame_detections in zip(camera_names, detected_points)
    if num_valid_detections(boards, fame_detections) == 0]
    
  assert len(cameras) == 0,\
    f"cameras {cameras} have no valid detections, check board config"
    

def check_image_lengths(cameras, filenames, image_names):
  for k, images in zip(cameras, filenames):
    assert len(images) == len(image_names),\
      f"mismatch between image names and camera {k}, "\
      f"got {len(images)} filenames expected {len(image_names)}"


def check_camera_images(camera_images):
  assert len(camera_images.cameras) == len(camera_images.filenames),\
    f"expected filenames to be a list of equal to number of cameras "\
    f"{len(camera_images.cameras)} vs. {len(camera_images.filenames)}"

  check_image_lengths(camera_images.cameras, camera_images.filenames, camera_images.image_names)
  if 'images' in camera_images is not None:
    check_image_lengths(camera_images.cameras, camera_images.images, camera_images.image_names)



class Workspace:

    def __init__(self, output_path, name="calibration"):

        self.name = name
        self.output_path = output_path

        self.calibrations = OrderedDict()
        self.detections = None
        self.image_quality = None

        self.boards = None
        self.board_colors = None

        self.filenames = None
        self.image_path = None
        self.names = struct()

        self.image_sizes = None
        self.images = None

        self.point_table = None
        self.pose_table = None

        self.pose_init = None

        self.log_handler = MemoryHandler()

    def add_camera_images(self, camera_images, j=cpu_count()):
        check_camera_images(camera_images)
        self.names = self.names._extend(
            camera=camera_images.cameras, image=camera_images.image_names)

        self.filenames = camera_images.filenames
        self.image_path = camera_images.image_path
        
        if 'images' in camera_images:
          self.images = camera_images.images
          self.image_size = map_list(common_image_size, self.images)
        else:
          self._load_images(j=j)

    def _load_images(self, j=cpu_count()):
        assert self.filenames is not None, "_load_images: no filenames set"

        info("Loading images..")
        self.images = image.detect.load_images(
            self.filenames, j=j, prefix=self.image_path)
        self.image_size = map_list(common_image_size, self.images)

        info(f"Loaded {self.sizes.image * self.sizes.camera} images")
        info(
            {k: image_size for k, image_size in zip(
                self.names.camera, self.image_size)})

              
    @property 
    def detections_file(self):
      return path.join(self.output_path, f"{self.name}.detections.pkl")


    def detect_boards(self, boards, load_cache=True, j=cpu_count()):
        assert self.boards is None, "detect_boards: boards already set"
        assert self.images is not None, "detect_boards: no images loaded, first use add_camera_images"

        board_names, self.boards = split_dict(boards)
        self.names = self.names._extend(board=board_names)
        self.board_colors = color_sets['set1']
        cache_key = self.fields("filenames", "boards", "image_sizes")

        self.detected_points = detect_boards_cached(self.boards, self.images, 
          self.detections_file, cache_key, load_cache, j=j)

        self.point_table = tables.make_point_table(self.detected_points, self.boards)
        info("Detected point counts:")
        tables.table_info(self.point_table.valid, self.names)

        info("Estimating image quality")
        self.image_quality = np.array(estimate_quality(self.images, self.detected_points))



    def sample_images(self, max_images=None):
      assert self.pose_init is not None, "sample_images: run initialise_pose first"
      
      if max_images is not None and max_images < self.sizes.image:
        indexes = sample_poses(self.pose_init.times.poses, max_images)
        info(f"filtering images down to {max_images}/{self.sizes.image} by furthest sampling")

        self.mask_frames(np.sort(indexes))


    def mask_frames(self, valid):
      if self.pose_init is not None:
        self.pose_init.times = self.pose_init.times._index_select(valid)

      if self.pose_table is not None:
        self.pose_table = self.pose_table._index_select(valid, axis=1)

      def mask_list(xs):
        arr = np.empty(len(xs), object)
        arr[:] = xs
        return arr[valid].tolist()

      def mask_lists(xss):
        return [mask_list(xs) for xs in xss]

      self.names.image = mask_list(self.names.image)
      self.filenames = mask_lists(self.filenames)
      self.detected_points = mask_lists(self.detected_points)

      self.image_quality = self.image_quality[:, valid]

      self.images = [mask_list(images) for images in self.images]
      self.point_table = self.point_table._index_select(valid, axis=1)


    def set_calibration(self, cameras):
      assert set(self.names.camera) == set(cameras.keys()),\
         f"set_calibration: cameras don't match"\
         f"{set(self.names.camera)} vs. {set(cameras.keys())}"

      self.cameras = [cameras[k] for k in self.names.camera]
      info("Camera initialization set externally:")
      for name, camera in zip(self.names.camera, self.cameras):
          info(f"{name} {camera}")
          info("")

    def calibrate_single(self, camera_model, fix_aspect=False, has_skew=False, max_images=None):
        assert self.detected_points is not None, "calibrate_single: no points found, first use detect_boards to find corner points"


        check_detections(self.names.camera, self.boards, self.detected_points)

        info("Calibrating single cameras..")
        self.cameras, errs = calibrate_cameras(
            self.boards,
            self.detected_points,
            self.image_size,
            model=camera_model,
            fix_aspect=fix_aspect,
            has_skew=has_skew,
            max_images=max_images,
        )

        for name, camera, err in zip(self.names.camera, self.cameras, errs):
            info(f"Calibrated {name}, with RMS={err:.2f}")
            info(camera)
            info("")

    



    def initialise_poses(self,  camera_poses=None, min_valid_cameras=1):
        assert self.cameras is not None, "initialise_poses: no cameras set, first use calibrate_single or set_cameras"
        self.pose_table = tables.make_pose_table(self.point_table, self.boards, self.cameras)

        info("Pose counts:")
        tables.table_info(self.pose_table.valid, self.names)

        visible_cameras = np.sum(np.any(self.pose_table.valid, axis=-1), axis=0) 
        valid = visible_cameras >= min_valid_cameras

        info(f"Found {valid.sum()}/{valid.shape[0]} valid frames")
        self.mask_frames(np.nonzero(valid)[0])


        self.pose_init = tables.initialise_poses(self.pose_table, 
          camera_poses=None if camera_poses is None else np.array([camera_poses[k] for k in self.names.camera])
        )




    def initialise_calibration(self, motion_model=StaticFrames):
        assert self.pose_init is not None, "filter_images: run initialise_pose first"

        calib = Calibration(
            ParamList(self.cameras, self.names.camera),
            ParamList(self.boards, self.names.board),
            self.point_table,
            PoseSet(self.pose_init.camera, self.names.camera),
            PoseSet(self.pose_init.board, self.names.board),
            motion_model.init(self.pose_init.times, self.names.image),
        )

        self.calibrations["initialisation"] = calib
        return calib

    def calibrate(self, name="calibration",
        camera_poses=True, motion=True, board_poses=True, 
        cameras=False, boards=False,
        loss='linear', tolerance=1e-4, num_adjustments=3,
        quantile=0.75, auto_scale=None, outlier_threshold=5.0)  -> Calibration:

        calib : Calibration = self.latest_calibration.enable(
            cameras=cameras, boards=boards, camera_poses=camera_poses,
            motion=motion, board_poses=board_poses)
            
        calib = calib.adjust_outliers(
          loss=loss, 
          tolerance=tolerance,
          num_adjustments=num_adjustments,
          select_outliers = select_threshold(quantile=quantile, factor=outlier_threshold),
          select_scale = select_threshold(quantile=quantile, factor=auto_scale) if auto_scale is not None else None
        )

        self.calibrations[name] = calib
        return calib

    @property
    def sizes(self):
        return self.names._map(len)

    @property
    def initialisation(self)  -> Calibration:
        return self.calibrations["initialisation"]

    @property
    def latest_calibration(self) -> Calibration:
        return list(self.calibrations.values())[-1]

    @property
    def log_entries(self):
        return self.log_handler.records

    def has_calibrations(self):
        return len(self.calibrations) > 0

    def get_calibrations(self):
        return self.calibrations

    def push_calibration(self, name, calib):
      if name in self.calibrations:
        raise KeyError(f"calibration {name} exists already {list(self.calibrations.keys())}")
      self.calibrations[name] = calib

    def get_camera_sets(self):
        if self.has_calibrations():
            return {k: calib.cameras for k, calib in self.calibrations.items()}

        if self.cameras is not None:
            return dict(initialisation=self.cameras)

    def export_json(self, master=None):
        master = master or self.names.camera[0]
        assert (
            master is None or master in self.names.camera
        ), f"master f{master} not found in cameras f{str(self.names.camera)}"

        calib = self.latest_calibration
        if master is not None:
            calib = calib.with_master(master)

        return export_json(calib, self.names, self.filenames, master=master)


    def export(self, filename=None, master=None):
      filename = filename or path.join(self.output_path, f"{self.name}.json")
      info(f"Exporting calibration to {filename}")

      data = self.export_json(master=master)
      with open(filename, 'w') as outfile:
        json.dump(to_dicts(data), outfile, indent=2)
        
    def dump(self, filename=None):
        filename = filename or path.join(self.output_path, f"{self.name}.pkl")

        info(f"Dumping state and history to {filename}")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        assert path.isfile(
            filename), f"Workspace.load: file does not exist {filename}"
        with open(filename, "rb") as file:
            ws = pickle.load(file)
            return ws

    def fields(self, *keys):
        return subset(self.__dict__, keys)

    def __getstate__(self):
        return self.fields(
            "calibrations",
            "detections",
            "boards",
            "board_colors",
            "filenames",
            "image_quality",
            "image_path",
            "names",
            "image_sizes",
            "point_table",
            "pose_table",
            "log_handler",
        )

    def __setstate__(self, d):
        for k, v in d.items():
            self.__dict__[k] = v

        self.images = None
