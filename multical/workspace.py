from collections import OrderedDict
import pathlib
from multical.motion import StaticFrames
from multiprocessing import cpu_count

from multical.optimization.parameters import ParamList
from multical.optimization.pose_set import PoseSet
from multical import config

from os import path
from multical.io import export, try_load_detections, write_detections
from multical.image.detect import common_image_size

from multical.optimization.calibration import Calibration
from structs.struct import map_list, split_dict, struct, subset
from . import tables, image
from .camera import calibrate_cameras

from .io.logging import MemoryHandler, info
from .display import make_palette

import pickle


class Workspace:
    def __init__(self, output_path, name="calibration"):

        self.name = name
        self.output_path = output_path

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

        self.log_handler = MemoryHandler()

    def load_camera_images(self, camera_images, j=cpu_count()):
        self.names = self.names._extend(
            camera=camera_images.cameras, image=camera_images.image_names)

        self.filenames = camera_images.filenames
        self.image_path = camera_images.image_path

        info("Loading images..")
        self.images = image.detect.load_images(
            self.filenames, j=j, prefix=self.image_path)
        self.image_size = map_list(common_image_size, self.images)

        info(f"Loaded {self.sizes.image * self.sizes.camera} images")
        info(
            {k: image_size for k, image_size in zip(
                self.names.camera, self.image_size)})

    @property
    def temp_folder(self):
      folder = pathlib.Path(self.output_path).joinpath("." + self.name)
      folder.mkdir(parents=True, exist_ok=True)
      return folder.name

    @property 
    def detections_file(self):
      return path.join(self.temp_folder, "detections.pkl")


    def detect_boards(self, boards, load_cache=True, j=cpu_count()):
        assert self.boards is None

        board_names, self.boards = split_dict(boards)
        self.names = self.names._extend(board=board_names)
        self.board_colors = make_palette(len(boards))
        cache_key = self.fields("filenames", "boards", "image_sizes")

        self.detected_points = (try_load_detections(
            self.detections_file, cache_key) if load_cache else None)

        if self.detected_points is None:
            info("Detecting boards..")
            self.detected_points = image.detect.detect_images(self.boards, self.images, j=j)

            info(f"Writing detection cache to {self.detections_file}")
            write_detections(self.detections_file, self.detected_points, cache_key)

        self.point_table = tables.make_point_table(self.detected_points, self.boards)
        info("Detected point counts:")
        tables.table_info(self.point_table.valid, self.names)

    def calibrate_single(self, camera_model, fix_aspect=False, has_skew=False, max_images=None):
        assert self.detected_points is not None

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
            motion_model.init(pose_init.times, self.names.image),
        )

        # calib = calib.reject_outliers_quantile(0.75, 5)
        calib.report(f"Initialisation")

        self.calibrations["initialisation"] = calib
        return calib

    def calibrate(self, name,
        camera_poses=True, motion=True, board_poses=True, 
        cameras=False, boards=False, **opt_args):

        calib = self.latest_calibration.enable(
            cameras=cameras, boards=boards, camera_poses=camera_poses,
            motion=motion, board_poses=board_poses)
            
        calib = calib.adjust_outliers(**opt_args)

        self.calibrations[name] = calib
        return calib

    @property
    def sizes(self):
        return self.names._map(len)

    @property
    def initialisation(self):
        return self.calibrations["initialisation"]

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
            return {k: calib.cameras for k, calib in self.calibrations.items()}

        if self.cameras is not None:
            return dict(initialisation=self.cameras)

    def export(self, filename=None, master=None):

        filename = filename or path.join(self.output_path, f"{self.name}.json")
        info(f"Exporting calibration to {filename}")

        master = master or self.names.camera[0]
        assert (
            master is None or master in self.names.camera
        ), f"master f{master} not found in cameras f{str(self.names.camera)}"

        calib = self.latest_calibration
        if master is not None:
            calib = calib.with_master(master)

        export(filename, calib, self.names, master=master)

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
