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


def initialise(args):

    ws = workspace.Workspace()
    paths = get_paths(args.outputs.output_path or args.inputs.image_path)

    setup_logging(args.runtime.log_level, [
                  ws.log_handler], log_file=paths.log_file)
    info(pformat_struct(args))

    boards = find_board_config(
        args.inputs.image_path, board_file=args.inputs.boards)
    camera_images = find_camera_images(args.inputs)

    ws.load_camera_images(camera_images, j=args.runtime.num_threads)
    ws.detect_boards(boards, cache_file=paths.detection_cache,
                     load_cache=not args.runtime.no_cache, j=args.runtime.num_threads)

    ws.calibrate_single(args.camera.distortion_model, fix_aspect=args.camera.fix_aspect,
                        has_skew=args.camera.allow_skew, max_images=args.camera.limit_intrinsic)

    ws.initialise_poses(
        motion_model=get_motion_model(args.camera.motion_model))
    return ws, paths


def optimize(ws, args):

    opt = args.optimizer
    params = args.parameters

    outliers = select_threshold(quantile=0.75, factor=opt.outlier_threshold)
    auto_scale = select_threshold(quantile=0.75, factor=opt.auto_scale)\
        if opt.auto_scale is not None else None

    ws.calibrate("calibration", loss=opt.loss,
                 boards=params.adjust_board,
                 cameras=not params.fix_intrinsic,
                 camera_poses=not params.fix_camera_poses,
                 board_poses=not params.fix_board_poses,
                 motion=not params.fix_motion,
                 auto_scale=auto_scale, outliers=outliers)
