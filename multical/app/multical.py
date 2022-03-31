from dataclasses import dataclass
from multical.config.arguments import run_with
from multiprocessing import cpu_count
from typing import  Union


from multical.app.boards import Boards
from multical.app.calibrate import Calibrate
from multical.app.intrinsic import Intrinsic
from multical.app.vis import Vis


@dataclass
class Multical:
  """multical - multi camera calibration 
  - calibrate: multi-camera calibration
  - intrinsic: calibrate separate intrinsic parameters
  - boards: generate/visualize board images, test detections
  - vis: visualize results of a calibration 
  """ 
  command : Union[Calibrate, Intrinsic, Boards, Vis]
   
  def execute(self):
    return self.command.execute()


def cli():
  run_with(Multical)

if __name__ == '__main__':
  cli()
