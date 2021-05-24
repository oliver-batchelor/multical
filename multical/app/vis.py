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
      visualize(self)


def fix_qt():
  # work around Qt in OpenCV 
  for k, v in os.environ.items():
      if k.startswith("QT_") and "cv2" in v:
          del os.environ[k]


def visualize_ws(ws):
    try:
      fix_qt()
      
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
    ws._load_images()

    visualize_ws(ws)

if __name__ == '__main__':
    run_with(Vis)
