from multical.workspace import detect_boards_cached
from os import path
import pathlib
from multical.app.calibrate import calibrate
from multical.config.runtime import find_board_config, find_camera_images
from multical.image.detect import common_image_size
from multical.threading import map_lists
from multical.io.logging import setup_logging
from multical.io.logging import info

from structs.struct import map_none, map_list
from multical import image

from structs.numpy import struct, shape

from multical.config.arguments import *

@dataclass
class Intrinsic:
  """Run separate intrinsic calibration for set of cameras"""
  paths     : PathOpts
  camera    : CameraOpts
  runtime   : RuntimeOpts

  def execute(self):
      calibrate_intrinsic(self)


def setup_paths(paths):
  output_path = paths.image_path or paths.output_path 
  temp_folder = pathlib.Path(output_path).joinpath("." + paths.name)
  temp_folder.mkdir(exist_ok=True, parents=True)
  return struct(
    output = output_path,
    temp=str(temp_folder),
    log_file=str(temp_folder.joinpath("log.txt")),
    detections=str(temp_folder.joinpath("detections.pkl"))
  )


def calibrate_intrinsic(args):
    paths=setup_paths(args.paths)

    setup_logging(args.runtime.log_level, [], log_file=paths.log_file)
    info(args) 

    boards = find_board_config(args.paths.image_path, args.paths.boards)

    camera_images = find_camera_images(args.paths.image_path, 
      args.paths.cameras, args.paths.camera_pattern, matching=False)


    image_counts = {k:len(files) for k, files in zip(camera_images.cameras, camera_images.filenames)}
    info("Found camera directories with images {}".format(image_counts))


    info("Loading images..")
    images = image.detect.load_images(camera_images.filenames,  
      prefix=camera_images.image_path, j=args.runtime.num_threads)
    image_sizes = map_list(common_image_size, images)

    info({k:image_size for k, image_size in zip(camera_images.cameras, image_sizes)})

    cache_key = struct(boards=boards, image_sizes=image_sizes, filenames=camera_images.filenames)

    detected_points = detect_boards_cached(boards, images, 
        paths.detections, cache_key, j=args.runtime.num_threads)
    
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
  run_with(Intrinsic)
