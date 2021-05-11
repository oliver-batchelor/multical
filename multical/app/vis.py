from os import path
import numpy as np

from multical.workspace import Workspace

from multical.io.logging import error
from multical.io.logging import setup_logging

from multical.config.arguments import *


@dataclass
class Vis:
    workspace_file : str 

    def execute(self):
      pass



def visualize_ws(ws):
    try:
      from multical.interface import visualizer
      visualizer.visualize(ws)

    except ImportError as err:     
      error(err)
      error("qtpy and pyvista are necessary to run the visualizer, install with 'pip install qtpy pyvista-qt'")


def visualize(args): 
    np.set_printoptions(precision=4, suppress=True)

    filename = args.workspace_file
    if path.isdir(filename):
      filename = path.join(filename, "calibration.pkl")
      
    ws = Workspace.load(filename)
    setup_logging('INFO', [ws.log_handler])
    ws.load_images()

    visualize_ws(ws)


if __name__ == '__main__':
  args = parse_with(Vis)
  visualize(args)