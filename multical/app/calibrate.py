
from .vis import visualize_ws

from structs.struct import struct, map_none, to_structs
import numpy as np

from multical.config import *
from dataclasses import dataclass

@dataclass
class Calibrate:
    """Run camera calibration"""
    inputs  : Inputs 
    outputs : Outputs 
    camera  : Camera 
    parameters : Parameters 
    runtime    : Runtime 
    optimizer  : Optimizer 
    vis : bool = False        # Visualize result after calibration

    def execute(self):
        calibrate(self)


def calibrate(args): 
  np.set_printoptions(precision=4, suppress=True)

  ws, paths = initialise(to_structs(args))
  optimize(ws, args)

  ws.export(paths.export_file)
  ws.dump(paths.workspace_file)

  if args.vis:
    visualize_ws(ws)



if __name__ == '__main__':
  run_with(Calibrate)
