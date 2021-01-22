
import logging
import math
from os import path
import pathlib
import os
from sys import stdout
import numpy as np
import argparse
import cv2

from multical import tables, board, workspace

from structs.struct import struct, transpose_lists, map_none
from structs.numpy import shape, Table
from pprint import pprint

from multical.interface import visualize
from logging import warning, info
import textwrap

class IndentFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(IndentFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
      message = record.msg
      record.msg = ''
      header = super(IndentFormatter, self).format(record)
      msg = textwrap.indent(message, ' ' * len(header)).strip()
      return header + msg


def setup_logging(output_path, console_level='INFO'):

    log_file = path.join(output_path, "calibration.log")


    stream_handler = logging.StreamHandler(stream=stdout)
    stream_handler.setLevel(getattr(logging, console_level))
    stream_handler.setFormatter(IndentFormatter('%(message)s'))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(IndentFormatter('%(levelname)s - %(message)s'))
    
    handlers = [
      stream_handler,
      file_handler
    ]

    logging.basicConfig(handlers=handlers, level=logging.DEBUG)
    info(f"Logging to {log_file}")
 


def main(): 
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='input image path')

    parser.add_argument('--save', default=None, help='save calibration as json default: input/calibration.json')

    parser.add_argument('--j', default=len(os.sched_getaffinity(0)), type=int, help='concurrent jobs')
    parser.add_argument('--show', default=False, action="store_true", help='show detections')

    parser.add_argument('--cameras', default=None, help="comma separated list of camera directories")
    
    parser.add_argument('--iter', default=3, help="iterations of bundle adjustment/outlier rejection")

    parser.add_argument('--fix_aspect', default=False, action="store_true", help='set same focal length ')
    parser.add_argument('--model', default="standard", help='camera model (standard|rational|thin_prism|tilted)')
    parser.add_argument('--boards', help='configuration file (YAML) for calibration boards')
 
    parser.add_argument('--intrinsic_images', default=None, help='number of images to use for initial intrinsic calibration default (unlimited)')
 
    parser.add_argument('--log_level', default='INFO', help='logging level for output to terminal')
    parser.add_argument('--output_path', default=None, help='specify output path, default (image_path)')



    args = parser.parse_args()
    output_path = args.output_path or args.image_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    setup_logging(output_path, args.log_level)
 
    ws = workspace.Workspace()
    info(args) 

    boards = board.load_config(args.boards)
    info("Using boards:")
    for name, b in boards.items():
      info(f"{name} {b}")

    cameras = map_none(str.split, args.cameras, ",")
    
    ws.find_images(args.image_path, cameras)
    ws.load_detect(boards, j=args.j)

    ws.calibrate_single(args.model, args.fix_aspect, args.intrinsic_images)

    # visualize(ws)

    assert False    

    point_table = tables.make_point_table(loaded.points, boards)
    calib = Calibration.initialise(cameras, boards, point_table)

    calib.report("initialisation")
    # calib = calib.reject_outliers_quantile(quantile=0.75, factor=4).bundle_adjust()   

    # calib = calib.reject_outliers_quantile(quantile=0.75, factor=2).enable_intrinsics().bundle_adjust()    
    # calib.report("optimised")

    # calib = calib.enable_board().enable_intrinsics()
    calib = calib.adjust_outliers(iterations=args.iter, quantile=0.75, factor=3, loss='soft_l1')
    calib.report("optimised")

    # calib = calib.enable_board().enable_intrinsics()
    # calib = calib.adjust_outliers(iterations=args.iter, quantile=0.75, factor=3, loss='soft_l1')
    # calib.report("optimised(intrinsics, board)")

    vis = visualize(calib, loaded.images, camera_names, image_names)
    

    # save = args.save or path.join(args.input, "calibration.json")
    # print("writing calibration to:", save)
    
    # io.export(save, calib, camera_names, image_names)
    # calib.plot_errors()
    # calib.display(loaded.images)


if __name__ == '__main__':
    main()
