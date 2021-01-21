
import logging
import math
from os import path
import os
import numpy as np
import argparse
import cv2

from functools import partial
from multiprocessing.pool import ThreadPool

from multical import tables, image, io, board, display, workspace

from multical.camera import Camera

from structs.struct import struct, transpose_lists, map_none
from structs.numpy import shape, Table
from pprint import pprint

from multical.interface import visualize
from logging import warning, info


def calibrate_cameras(boards, points, image_sizes, **kwargs):
  pool = ThreadPool()
  f = partial(Camera.calibrate, boards, **kwargs) 
  return transpose_lists(pool.starmap(f, zip(points, image_sizes)))


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
 
    parser.add_argument('--log_level', default='DEBUG', help='logging level for output to terminal')
 
    args = parser.parse_args()
    info(args) 

    logging.basicConfig(filename='example.log', filemode='w', level=getattr(logging, args.log_level))

    boards = board.load_config(args.boards)
    info("Using boards:")
    for name, b in boards.items():
      info(name, b)

    ws = workspace.Workspace()

    cameras = map_none(str.split, args.cameras, ",")
    
    ws.find_images(args.image_path, cameras)
    ws.load_detect(boards, j=args.j)

    ws.calibrate_single(args.model, args.fix_aspect)

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
