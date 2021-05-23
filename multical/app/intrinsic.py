from multical.io.export_calib import export_single
from multical.camera import calibrate_cameras
from multical.workspace import detect_boards_cached
from os import path
import pathlib
from multical.config.runtime import find_board_config, find_camera_images
from multical.image.detect import common_image_size
from multical.io.logging import setup_logging
from multical.io.logging import info

from structs.struct import  map_list, pformat_struct, split_dict
from multical import image

from structs.numpy import struct, shape

from multical.config.arguments import *

@dataclass
class Intrinsic:
  """Run separate intrinsic calibration for set of cameras"""
  paths     : PathOpts = PathOpts(name="intrinsic")
  camera    : CameraOpts = CameraOpts()
  runtime   : RuntimeOpts = RuntimeOpts()

  def execute(self):
      calibrate_intrinsic(self)


def setup_paths(paths):
  output_path = paths.image_path or paths.output_path 
  temp_folder = pathlib.Path(output_path).joinpath("." + paths.name)
  temp_folder.mkdir(exist_ok=True, parents=True)

  return struct(
    output = output_path,
    temp=str(temp_folder),

    calibration_file=path.join(output_path, f"{paths.name}.json"),
    log_file=str(temp_folder.joinpath("log.txt")),
    detections=str(temp_folder.joinpath("detections.pkl"))
  )


def calibrate_intrinsic(args):
    paths=setup_paths(args.paths)

    setup_logging(args.runtime.log_level, [], log_file=paths.log_file)
    info(pformat_struct(args)) 

    image_path = os.path.expanduser(args.paths.image_path)
    info(f"Finding images in {image_path}")

    camera_images = find_camera_images(image_path, 
      args.paths.cameras, args.paths.camera_pattern, matching=False)

    image_counts = {k:len(files) for k, files in zip(camera_images.cameras, camera_images.filenames)}
    info("Found camera directories with images {}".format(image_counts))

    board_names, boards = split_dict(find_board_config(image_path, args.paths.boards))

    info("Loading images..")
    images = image.detect.load_images(camera_images.filenames,  
      prefix=camera_images.image_path, j=args.runtime.num_threads)
    image_sizes = map_list(common_image_size, images)


    info({k:image_size for k, image_size in zip(camera_images.cameras, image_sizes)})
    cache_key = struct(boards=boards, image_sizes=image_sizes, filenames=camera_images.filenames)

    detected_points = detect_boards_cached(boards, images, 
        paths.detections, cache_key, j=args.runtime.num_threads)

    cameras, errs = calibrate_cameras(boards, detected_points, image_sizes,  
      model=args.camera.distortion_model, fix_aspect= args.camera.fix_aspect, 
      has_skew=args.camera.allow_skew, max_images= args.camera.limit_intrinsic)
     
    for name, camera, err in zip(camera_images.cameras, cameras, errs):
        info(f"Calibrated {name}, with RMS={err:.2f}")
        info(camera)
        info("")

    info(f"Writing single calibrations to {paths.calibration_file}")
    export_single(paths.calibration_file, cameras, camera_images.cameras, camera_images.filenames)


if __name__ == '__main__':
  run_with(Intrinsic)
