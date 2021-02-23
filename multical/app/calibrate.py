import logging
import math
from multical.motion.static_frames import StaticFrames
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

from multical.io.logging import warning, info

from .arguments import add_calibration_args, parse_arguments

def get_paths(args):
    np.set_printoptions(precision=4, suppress=True)

    output_path = args.output_path or args.image_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    return struct(
      log_file = path.join(output_path, f"{args.name}.log"),
      export_file = path.join(output_path, f"{args.name}.json"),

      detection_cache = path.join(output_path, f"detections.pkl"),
      workspace_file = path.join(output_path, f"{args.name}.pkl")
    )


def initialise(args, paths):
    ws = workspace.Workspace()
 
    setup_logging(args.log_level, [ws.log_handler], log_file=paths.log_file)
    info(args) 

    board_file = args.boards or path.join(args.image_path, "boards.yaml")
    boards = board.load_config(board_file)
    info("Using boards:")
    for name, b in boards.items():
      info(f"{name} {b}")

    cameras = map_none(str.split, args.cameras, ",")
    
    ws.find_images(args.image_path, cameras, master = args.master)
    ws.load_images(j=args.j)
    ws.detect_boards(boards, j=args.j, cache_file=paths.detection_cache, load_cache=not args.no_cache)
    
    ws.calibrate_single(args.model, fix_aspect=args.fix_aspect, 
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

def optimize(args, ws):

    outliers =  select_threshold(quantile=0.75, factor=6)
    auto_scale = select_threshold(quantile=0.75, factor=2)


    ws.calibrate("calibration", loss=args.loss,  
      cameras=True, 
      boards=args.adjust_board,
      auto_scale=auto_scale, outliers=outliers)

    ws.calibrate("final", loss=args.loss,  
      tolerance = 1e-5, max_iterations=30, 
      num_adjustments=1,
      intrinsics=True, 
      board=True,
      auto_scale=auto_scale, outliers=outliers)  

def visualize(ws):
    try:
      from multical.interface import visualizer
      visualizer.visualize(ws)

    except ImportError as error:     
      print(error.__class__.__name__ + ": " + error.message)
      print("Pyside2 is necessary to run the visualizer")


def calibrate(args): 
    paths = get_paths(args)

    ws = initialise(args, paths)
    optimize(args, ws)

    ws.export(paths.export_file)
    ws.dump(paths.workspace_file)

    if args.show:
      visualize(ws)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_calibration_args(parser)
    args = parser.parse_args()

    calibrate(args)
