import dataclasses

from structs.struct import pformat_struct, pprint_struct, to_structs
from multical.motion.static_frames import StaticFrames
from multical.motion.rolling_frames import RollingFrames
from multical.optimization.calibration import select_threshold
from multical.io.logging import setup_logging
from multical import board, workspace
from multical.io.logging import info

from .runtime import find_board_config, find_camera_images, get_paths
from .arguments import *


def get_motion_model(motion_model):
    if motion_model == "rolling":
        return RollingFrames
    elif motion_model == "static":
        return StaticFrames
    else:
        assert False, f"unknown motion model {motion_model}, (static|rolling)"


def initialise_with_images(ws, boards, camera_images, 
  camera_opts : CameraOpts = CameraOpts(), runtime : RuntimeOpts = RuntimeOpts()):

    ws.load_camera_images(camera_images, j=runtime.num_threads)
    ws.detect_boards(boards, load_cache=not runtime.no_cache, j=runtime.num_threads)

    ws.calibrate_single(camera_opts.distortion_model, fix_aspect=camera_opts.fix_aspect,
                        has_skew=camera_opts.allow_skew, max_images=camera_opts.limit_intrinsic)

    ws.initialise_poses(
        motion_model=get_motion_model(camera_opts.motion_model))
    return ws


def optimize(ws, opt : OptimizerOpts = OptimizerOpts()):
  outliers = select_threshold(quantile=0.75, factor=opt.outlier_threshold)
  auto_scale = select_threshold(quantile=0.75, factor=opt.auto_scale)\
      if opt.auto_scale is not None else None

  ws.calibrate("calibration", loss=opt.loss,
    boards=opt.adjust_board,
    cameras=not opt.fix_intrinsic,
    camera_poses=not opt.fix_camera_poses,
    board_poses=not opt.fix_board_poses,
    motion=not opt.fix_motion,
    auto_scale=auto_scale, outliers=outliers)

  return ws