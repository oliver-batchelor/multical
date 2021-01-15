
import math
from os import path
import os
import numpy as np
import argparse
import cv2

from functools import partial
from multiprocessing.pool import ThreadPool

from multical import tables, image, io
from multical.optimization.calibration import Calibration

from multical.camera import Camera
from multical.board import CharucoBoard

from structs.struct import struct, transpose_lists
from structs.numpy import shape, Table
from pprint import pprint

from multical.interface import visualize


def calibrate_cameras(board, points, image_sizes, **kwargs):
  pool = ThreadPool()
  f = partial(Camera.calibrate, board, **kwargs) 
  return transpose_lists(pool.starmap(f, zip(points, image_sizes)))


def main(): 
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path')

    parser.add_argument('--save', default=None, help='save calibration as json default: input/calibration.json')

    parser.add_argument('--j', default=len(os.sched_getaffinity(0)), type=int, help='concurrent jobs')
    parser.add_argument('--show', default=False, action="store_true", help='show detections')

    parser.add_argument('--cameras', default=None, help="comma separated list of camera directories")
    
    parser.add_argument('--iter', default=3, help="iterations of bundle adjustment/outlier rejection")

    parser.add_argument('--fix_aspect', default=False, action="store_true", help='set same focal length ')
    parser.add_argument('--model', default="standard", help='camera model (standard|rational|thin_prism|tilted)')
 
    args = parser.parse_args()
    print(args) 

    cameras = args.cameras.split(",") if args.cameras is not None else None
    camera_names, image_names, filenames = image.find.find_images(args.input, cameras)
    print("Found camera directories {} with {} matching images".format(str(camera_names), len(image_names)))

    board = CharucoBoard.create(size=(16, 22), square_length=0.025, marker_length=0.01875, 
      aruco_dict='4X4_1000', min_rows=3, min_points=20)

    print("Detecting patterns..")
    loaded = image.detect.detect_images(board, filenames, j=args.j, prefix=args.input)   

    print("Calibrating cameras..")
    cameras, errs = calibrate_cameras(board, loaded.points, loaded.image_size, model=args.model, fix_aspect=args.fix_aspect)

    for name, camera, err in zip(camera_names, cameras, errs):
      print(f"Calibrated {name}, with RMS={err:.2f}")
      print(camera)
      print("---------------")
    
    
    point_table = tables.make_point_table(loaded.points, board)
    calib = Calibration.initialise(cameras, board, point_table)

    calib.report("initialisation")

    
    calib = calib.bundle_adjust(loss='huber')
    
    calib.reject_outliers_quantile(quantile=0.75, factor=2)
    calib.report("optimised")

    # calib = calib.enable_board().enable_intrinsics()
    # calib = calib.adjust_outliers(iterations=args.iter, upper_quartile=2.0)
    # calib.report("optimised")

    vis = visualize(calib, loaded.images, camera_names, image_names)
    

    # save = args.save or path.join(args.input, "calibration.json")
    # print("writing calibration to:", save)
    
    # io.export(save, calib, camera_names, image_names)
    # calib.plot_errors()
    # calib.display(loaded.images)


if __name__ == '__main__':
    main()
