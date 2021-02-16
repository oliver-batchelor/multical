import logging
import math
from multical.motion.rolling_frames import RollingFrames
from multical.optimization.calibration import select_threshold
from multical.io.logging import setup_logging
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

from multical.interface import visualize, visualizer
from multical.io.logging import warning, info


def main(): 
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='input image path')

    parser.add_argument('--save', default=None, help='save calibration as json default: input/calibration.json')

    parser.add_argument('--j', default=len(os.sched_getaffinity(0)), type=int, help='concurrent jobs')

    parser.add_argument('--cameras', default=None, help="comma separated list of camera directories")
    
    parser.add_argument('--iter', default=3, help="iterations of bundle adjustment/outlier rejection")
    parser.add_argument('--rolling', default=False, action="store_true", help='single frame rolling shutter estimation')


    parser.add_argument('--fix_aspect', default=False, action="store_true", help='set same focal length ')
    parser.add_argument('--model', default="standard", help='camera model (standard|rational|thin_prism|tilted)')
    parser.add_argument('--boards', default=None, help='configuration file (YAML) for calibration boards')
 
    parser.add_argument('--intrinsic_images', type=int, default=50, help='number of images to use for initial intrinsic calibration default (unlimited)')
 
    parser.add_argument('--log_level', default='INFO', help='logging level for output to terminal')
    parser.add_argument('--output_path', default=None, help='specify output path, default (image_path)')

    parser.add_argument('--loss', default='linear', help='loss function in optimizer (linear|soft_l1|huber|cauchy|arctan)')
    parser.add_argument('--no_cache', default=False, action='store_true', help="don't load detections from cache")

    parser.add_argument('--show', default=False, action="store_true", help='show calibration result')


    args = parser.parse_args()
    output_path = args.output_path or args.image_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    log_file = path.join(output_path, "calibration.log")
    export_file = path.join(output_path, "calibration.json")

    detection_cache = path.join(output_path, "detections.pkl")
    workspace_file = path.join(output_path, "workspace.pkl")
    
    ws = workspace.Workspace()
    
    setup_logging(args.log_level, [ws.log_handler], log_file=log_file)
 
    info(args) 

    board_file = args.boards or path.join(args.image_path, "boards.yaml")
    boards = board.load_config(board_file)
    info("Using boards:")
    for name, b in boards.items():
      info(f"{name} {b}")

    cameras = map_none(str.split, args.cameras, ",")
    
    ws.find_images(args.image_path, cameras)
    ws.load_images(j=args.j)
    ws.detect_boards(boards, j=args.j, cache_file=detection_cache, load_cache=not args.no_cache)
    
    ws.calibrate_single(args.model, args.fix_aspect, args.intrinsic_images)

    ws.initialise_poses(motion_model=RollingFrames)
    outliers =  select_threshold(quantile=0.75, factor=6)
    auto_scale = select_threshold(quantile=0.75, factor=2)
    # outliers = None


    ws.calibrate("calibration", loss=args.loss,  
      cameras=True, 
      # boards=True,
      auto_scale=auto_scale, outliers=outliers)


    # ws.calibrate("final", loss=args.loss,  
    #   tolerance = 1e-5, max_iterations=30, 
    #   num_adjustments=1,
    #   intrinsics=True, 
    #   # board=True,
    #   auto_scale=auto_scale, outliers=outliers)


    ws.export(export_file)
    ws.dump(workspace_file)

    # ws.calibrate("full", enable_intrinsics=True, enable_board=True, loss=args.loss)
    if args.show:
      visualizer.visualize(ws)


if __name__ == '__main__':
    main()
