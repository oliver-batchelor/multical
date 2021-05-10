from multical.config import find_board_config, find_camera_images, get_paths

from multical.motion.static_frames import StaticFrames
from multical.motion.rolling_frames import RollingFrames
from multical.optimization.calibration import select_threshold
from multical.io.logging import setup_logging
from multical import board, workspace
from multical.io.logging import info

from .arguments import add_calibration_args, parse_with
from .vis import visualize_ws

from structs.struct import struct, map_none
import numpy as np





def calibrate(args): 
    np.set_printoptions(precision=4, suppress=True)

    paths = get_paths(args.output_path or args.image_path)
 
    cameras = map_none(str.split, args.cameras, ",")
    camera_images = find_camera_images(args.image_path, cameras, args.camera_pattern, master = args.master)
 
    ws = initialise(args, paths, camera_images)
    optimize(args, ws)

    ws.export(paths.export_file)
    ws.dump(paths.workspace_file)

    if args.vis:
      visualize_ws(ws)


if __name__ == '__main__':
    args = parse_with(add_calibration_args)
    calibrate(args)
