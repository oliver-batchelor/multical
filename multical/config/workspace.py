from multical.motion.static_frames import StaticFrames
from multical.motion.rolling_frames import RollingFrames
from multical.optimization.calibration import select_threshold
from multical.io.logging import setup_logging
from multical import board, workspace
from multical.io.logging import info

from .runtime import find_board_config
from .arguments import *

def initialise(paths, camera_images, args):
    ws = workspace.Workspace()

    setup_logging(args.log_level, [ws.log_handler], log_file=paths.log_file)
    info(args)

    boards = find_board_config(args.image_path, board_file=args.boards)

    ws.load_images(camera_images, j=args.j)
    ws.detect_boards(boards, cache_file=paths.detection_cache,
                     load_cache=not args.no_cache, j=args.j)

    ws.calibrate_single(args.distortion_model, fix_aspect=args.fix_aspect,
                        has_skew=args.allow_skew, max_images=args.intrinsic_images)

    motion_model = None
    if args.motion_model == "rolling":
        motion_model = RollingFrames
    elif args.motion_model == "static":
        motion_model = StaticFrames
    else:
        assert False, f"unknown motion model {args.motion_model}, (static|rolling)"

    ws.initialise_poses(motion_model=motion_model)
    return ws


def optimize(ws, args):

    outliers = select_threshold(quantile=0.75, factor=args.outlier_threshold)
    auto_scale = select_threshold(quantile=0.75, factor=args.auto_scale)\
        if args.auto_scale is not None else None

    ws.calibrate("calibration", loss=args.loss,
                 boards=args.adjust_board,
                 cameras=not args.fix_intrinsic,
                 camera_poses=not args.fix_camera_poses,
                 board_poses=not args.fix_board_poses,
                 motion=not args.fix_motion,
                 auto_scale=auto_scale, outliers=outliers)