from multical.motion.static_frames import StaticFrames
from multical.motion.rolling_frames import RollingFrames
from multical.optimization.calibration import select_threshold
from multical.io.logging import setup_logging
from multical import board, workspace
from multical.io.logging import info

from .arguments import add_calibration_args, parse_with
from .show_result import visualize

from structs.struct import struct, map_none

from os import path
import pathlib
import numpy as np

from multical import tables, image

from .calibrate import get_paths, init_boards

def calibrate(args):
 
    paths = get_paths(args)

    setup_logging(args.log_level, [], log_file=paths.log_file)
    info(args) 

    boards = init_boards(args)

    cameras = map_none(str.split, args.cameras, ",")
    camera_paths = image.find.find_cameras(args.image_path, cameras, args.camera_pattern)

    camera_dirs, image_files = image.find.find_images_unmatched(camera_paths)
    image_counts = {k:len(files) for k, files in image_files.items()}
    info("Found camera directories with images {}".format(image_counts))


    # info("Loading images..")
    # self.images = image.detect.load_images(self.filenames, j=j, prefix=self.image_path)
    # self.image_size = map_list(common_image_size, self.images)

    # info(f"Loaded {self.sizes.image * self.sizes.camera} images")
    # info({k:image_size for k, image_size in zip(self.names.camera, self.image_size)})




    # ws.find_images_matching(args.image_path, cameras, args.camera_pattern, master = args.master)
 

    # ws.load_images(j=args.j)
    # ws.detect_boards(boards, cache_file=paths.detection_cache, 
    #   load_cache=not args.no_cache, j=args.j)
    
    # ws.calibrate_single(args.distortion_model, fix_aspect=args.fix_aspect, 
    #   has_skew=args.allow_skew, max_images=args.intrinsic_images)

    # motion_model = None
    # if args.motion_model == "rolling":
    #   motion_model = RollingFrames
    # elif args.motion_model == "static":
    #   motion_model = StaticFrames
    # else:
    #   assert False, f"unknown motion model {args.motion_model}, (static|rolling)"

    # ws.initialise_poses(motion_model=motion_model)
    # return ws




if __name__ == '__main__':
    args = parse_with(add_calibration_args)
    calibrate(args)
