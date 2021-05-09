
import argparse
from multiprocessing import cpu_count
import os

def add_paths_group(parser):
  paths = parser.add_argument_group('paths')

  parser.add_argument('--save', default=None, help='save calibration as json default: input/calibration.json')
  parser.add_argument('--name', default='calibration', help='name for this calibration (used to name output files)')
  parser.add_argument('--output_path', default=None, help='specify output path, default (image_path)')
  return paths

def add_image_paths_group(parser):
  image_paths = parser.add_argument_group('image paths')

  image_paths.add_argument('--camera_pattern', type=str, default=None, help='pattern to find images for explicitly provided cameras example "{camera}/extrinsic"')
  image_paths.add_argument('--cameras', default=None, help="explicit list of cameras (default: find subdirectories with matching images)")
  image_paths.add_argument('--intrinsic_images', type=int, default=50, help='limit images for initial intrinsic calibration (50)')
  return image_paths

def add_camera_group(parser):
  camera = parser.add_argument_group('camera settings')
  camera.add_argument('--fix_aspect', default=False, action="store_true", help='force cameras to have same focal length')
  camera.add_argument('--allow_skew', default=False, action="store_true", help='allow skew in intrinsic matrix')
  camera.add_argument('--distortion_model', default="standard", help='lens distortion model (standard|rational|thin_prism|tilted)')
  return camera

def add_misc_group(parser):
  misc = parser.add_argument_group('misc')

  misc.add_argument('--j', default=cpu_count(), type=int, help='concurrent jobs')
  misc.add_argument('--log_level', default='INFO', help='logging level for output to terminal')
  misc.add_argument('--no_cache', default=False, action='store_true', help="don't load detections from cache")

  return misc

def add_calibration_args(parser):

    parser.add_argument('image_path', help='input image path')
    add_paths_group(parser)
    add_image_paths_group(parser)

    camera = add_camera_group(parser)
    camera.add_argument('--master', default=None, help='use camera as master when exporting (default use first camera)')


    optimization = parser.add_argument_group('general optimization settings')
    
    optimization.add_argument('--iter', default=3, help="iterations of bundle adjustment/outlier rejection")
    optimization.add_argument('--boards', default=None, help='configuration file (YAML) for calibration boards')
    optimization.add_argument('--loss', default='linear', help='loss function in optimizer (linear|soft_l1|huber|cauchy|arctan)')

    optimization.add_argument('--outlier_threshold', default=5.0, help='threshold for outliers (factor of upper quartile of reprojection error)')
    optimization.add_argument('--auto_scale', default=None, help='threshold for auto_scale to reduce outlier influence (factor of upper quartile of reprojection error) - requires non-linear loss')

    enable = parser.add_argument_group('enable/disable optimization')

    enable.add_argument('--fix_intrinsic', default=False, action='store_true', help='fix intrinsics in optimization')
    enable.add_argument('--fix_camera_poses', default=False, action='store_true', help='fix camera pose optimization')
    enable.add_argument('--fix_board_poses', default=False, action='store_true', help='fix relative board positions (rely on initialization)')
    enable.add_argument('--fix_motion', default=False, action='store_true', help='fix motion optimization (rely on initialization)')
    
    enable.add_argument('--adjust_board', default=False, action='store_true', help='optimize non-planarity of board points')
    enable.add_argument('--motion_model', default="static", help='motion model (rolling|static)')
    
    misc = add_misc_group(parser)
    misc.add_argument('--vis', default=False, action="store_true", help='visualize result after calibration')

    parser.set_defaults(which='calibrate')


def add_intrinsic_args(parser):

    parser.add_argument('image_path', help='input image path')
    parser.add_argument('--boards', default=None, help='configuration file (YAML) for calibration boards')

    add_paths_group(parser)
    add_image_paths_group(parser)
    add_camera_group(parser)
    add_misc_group(parser)

    parser.set_defaults(which='intrinsic')


def add_boards_args(parser):
    parser.add_argument('boards',  help='configuration file (YAML) for calibration boards')
    parser.add_argument('--detect', default=None,  help='show detections from an image')

    parser.add_argument('--write', default=None,  help='directory to write board images instead of showing on screen')
    parser.add_argument('--pixels_mm', type=int, default=1,  help='pixels per mm')
    parser.add_argument('--margin_mm', type=int, default=20,  help='border width in mm')
    
    parser.add_argument('--paper_size_mm', type=str, default=None,  help='paper size in mm WxH or standard size A0..A4')

    parser.set_defaults(which='boards')


def add_vis_args(parser):
    parser.add_argument('workspace_file', help='workspace filename to load')

    parser.set_defaults(which='vis')


def parse_with(add_args, **kwargs):
    parser = argparse.ArgumentParser(**kwargs)
    add_args(parser)
    return parser.parse_args()

def parser():

    parser = argparse.ArgumentParser(prog='multical')
    subparsers = parser.add_subparsers()

    calibrate_parser = subparsers.add_parser('calibrate', help="run calibration process")
    add_calibration_args(calibrate_parser)

    intrinsic_parser = subparsers.add_parser('intrinsic', help="separate intrinsic calibration")
    add_intrinsic_args(intrinsic_parser)

    boards_parser = subparsers.add_parser('boards', help="check the board config matches expectations")
    add_boards_args(boards_parser)

    vis_parser = subparsers.add_parser('vis', help="visualise the result of a calibration")
    add_vis_args(vis_parser)

    return parser



