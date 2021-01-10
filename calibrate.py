import math
from os import path
import numpy as np
import argparse
import cv2


from multical import tables, image, io
from multical.optimization.calibration import Calibration

from multical.camera import Camera
from multical.board import CharucoBoard

from structs.struct import struct
from structs.numpy import shape, Table
from pprint import pprint

from multical.interface import visualize




def main(): 
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path')

    parser.add_argument('--save', default=None, help='save calibration as json default: input/calibration.json')

    parser.add_argument('--j', default=4, type=int, help='concurrent jobs')
    parser.add_argument('--show', default=False, action="store_true", help='show detections')

    parser.add_argument('--min_corners', default=5, type=int, help='minimum visible points to use estimated pose')
    parser.add_argument('--cameras', default=None, help="comma separated list of camera directories")
    
    parser.add_argument('--iter', default=3, help="iterations of bundle adjustment/outlier rejection")

    parser.add_argument('--fix_aspect', default=False, action="store_true", help='set same focal length ')
    parser.add_argument('--model', default="standard", help='camera model (standard|rational|thin_prism|tilted)')


    args = parser.parse_args()
    print(args)

    cameras = args.cameras.split(",") if args.cameras is not None else None
    camera_names, image_names, filenames = image.find.find_images(args.input, cameras)
    print("Found camera directories {} with {} matching images".format(str(camera_names), len(image_names)))


    board = CharucoBoard.create(size=(16, 22), square_length=0.025, 
                     marker_length=0.01875, aruco_dict='4X4_1000')


    loaded = image.detect.detect_images(board, camera_names, filenames, j=args.j, prefix=args.input)   


    cameras = []   
    for name, points, image_size in zip(camera_names, loaded.points, loaded.image_size):
      camera, err = Camera.calibrate(board, points, image_size=image_size, model=args.model, fix_aspect=args.fix_aspect) 
      cameras.append(camera)

      print(f"Calibrated {name}, with RMS={err:.2f}")
      print(camera)
      print("---------------")
    
    
    point_table = tables.make_point_table(loaded.points, board)
    calib = Calibration.initialise(cameras, board, point_table,  min_corners=20)

    calib.report("initialisation")


    vis = visualize(calib, loaded.images, camera_names, image_names)
    
    # calib = calib.bundle_adjust()
    # calib.report("optimised")

    # calib = calib.enable_board().enable_intrinsics().bundle_adjust()
    # calib = calib.adjust_outliers(iterations=args.iter, quantile=0.99)
    # calib.report("optimised")

    # save = args.save or path.join(args.input, "calibration.json")
    # print("writing calibration to:", save)
    
    # io.export(save, calib, camera_names, image_names)
    # calib.plot_errors()
    # calib.display(loaded.images)


if __name__ == '__main__':
    main()
