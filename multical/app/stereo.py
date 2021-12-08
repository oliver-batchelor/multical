
from multical.camera import stereo_calibrate
from multical.io.logging import setup_logging
from multical.optimization.pose_set import pose_str
from multical.tables import matching_points

import numpy as np

from multical.config import *
from dataclasses import dataclass

@dataclass
class Calibrate:
    """Run camera calibration"""
    paths  : PathOpts 
    camera  : CameraOpts
    runtime    : RuntimeOpts
    optimizer  : OptimizerOpts 
    vis : bool = False        # Visualize result after calibration

    def execute(self):
        calibrate(self)


def calibrate(args): 
  np.set_printoptions(precision=4, suppress=True)

  # Use image path if not explicity specified
  output_path = args.paths.image_path or args.paths.output_path 

  ws = workspace.Workspace(output_path, args.paths.name)
  setup_logging(args.runtime.log_level, [ws.log_handler], log_file=path.join(output_path, f"{args.paths.name}.txt"))

  boards = find_board_config(args.paths.image_path, board_file=args.paths.boards)

  camera_images = find_camera_images(args.paths.image_path, 
    args.paths.cameras, args.paths.camera_pattern, limit=args.paths.limit_images)

  initialise_with_images(ws, boards, camera_images, args.camera, args.runtime)

  

  for i in range(len(ws.cameras)):
    for j in range(i + 1, len(ws.cameras)):
        matching = matching_points(ws.point_table, ws.boards[0], i, j)

        names = (ws.names.camera[i], ws.names.camera[j])
        info(f"stereo_calibrate cameras {names}")
        left, right, pose, err = stereo_calibrate((ws.cameras[i], ws.cameras[j]), matching)

        info(f"RMS={err:.2f} {pose_str(pose)}")

        info(f"{names[0]}: {left}")
        info(f"{names[1]}: {right}")
    


if __name__ == '__main__':
  run_with(Calibrate)
