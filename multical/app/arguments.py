
import argparse
import os


def add_calibration_args(parser):

    parser.add_argument('image_path', help='input image path')

    parser.add_argument('--save', default=None, help='save calibration as json default: input/calibration.json')
    parser.add_argument('--name', default='calibration', help='name for this calibration (used to name output files)')


    parser.add_argument('--j', default=len(os.sched_getaffinity(0)), type=int, help='concurrent jobs')
    parser.add_argument('--cameras', default=None, help="comma separated list of camera directories")
    
    parser.add_argument('--iter', default=3, help="iterations of bundle adjustment/outlier rejection")
    parser.add_argument('--motion_model', default="static", help='motion model (rolling|static)')


    parser.add_argument('--fix_aspect', default=False, action="store_true", help='set same focal length ')
    parser.add_argument('--allow_skew', default=False, action="store_true", help='allow skew in intrinsic matrix')
    
    parser.add_argument('--master', default=None, help='use camera as master when exporting')

    
    parser.add_argument('--model', default="standard", help='camera model (standard|rational|thin_prism|tilted)')
    parser.add_argument('--boards', default=None, help='configuration file (YAML) for calibration boards')
 
    parser.add_argument('--intrinsic_images', type=int, default=50, help='number of images to use for initial intrinsic calibration default (unlimited)')
 
    parser.add_argument('--log_level', default='INFO', help='logging level for output to terminal')
    parser.add_argument('--output_path', default=None, help='specify output path, default (image_path)')

    parser.add_argument('--loss', default='linear', help='loss function in optimizer (linear|soft_l1|huber|cauchy|arctan)')
    parser.add_argument('--no_cache', default=False, action='store_true', help="don't load detections from cache")

    parser.add_argument('--show', default=False, action="store_true", help='show calibration result')

def add_boards_args(parser):
    parser.add_argument('boards',  help='configuration file (YAML) for calibration boards')
    parser.add_argument('--detect', default=None,  help='show detections from an image')


def add_show_args(parser):
    parser.add_argument('workspace_file', help='workspace filename to load')


def parse_with(add_args, **kwargs):
    parser = argparse.ArgumentParser(**kwargs)
    add_args(parser)
    return parser.parse_args()

def parse_arguments():

    parser = argparse.ArgumentParser(prog='multical')
    subparsers = parser.add_subparsers(required=True)

    calibrate_parser = subparsers.add_parser('calibrate')
    add_calibration_args(calibrate_parser)

    boards_parser = subparsers.add_parser('check_boards')
    add_boards_args(boards_parser)

    show_parser = subparsers.add_parser('show_result')
    add_show_args(show_parser)

    return parser.parse_args()


