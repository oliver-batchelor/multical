
from .vis import visualize_ws

from structs.struct import struct, map_none
import numpy as np

from multical.config import *
from dataclasses import dataclass

@dataclass
class Calibrate:
    """Run camera calibration"""
    inputs  : Inputs = Inputs()
    outputs : Outputs = Outputs()
    camera  : Camera = Camera()
    parameters : Parameters = Parameters()
    runtime    : Runtime = Runtime()
    optimizer  : Optimizer = Optimizer()
    vis : bool = False        # Visualize result after calibration

    def execute(self):
        calibrate(self)


def calibrate(args): 
    np.set_printoptions(precision=4, suppress=True)

    paths = get_paths(args.output_path or args.image_path)
 
    cameras = map_none(str.split, args.cameras, ",")
    camera_images = find_camera_images(args.image_path, cameras, args.camera_pattern, master = args.master)
 
    ws = initialise(args, paths, camera_images)
    optimize(args, ws)

    ws.export(paths.export_file)
    ws.dump(paths.workspace_file)

    if args.vis:
      visualize_ws(ws)


if __name__ == '__main__':
    args = parse_with(Calibrate)
    calibrate(args)
