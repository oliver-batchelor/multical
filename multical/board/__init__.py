from dataclasses import dataclass
from os import path
from .charuco import CharucoBoard
from .aprilgrid import AprilGrid
from .calico_config import load_calico


from typing import Tuple

from omegaconf.omegaconf import OmegaConf, MISSING
from structs.struct import struct

from multical.io.logging import debug, info, error

@dataclass 
class CharucoConfig:
  _type_: str = "charuco"
  size : Tuple[int, int] = MISSING

  square_length : float = MISSING
  marker_length : float = MISSING
  
  aruco_dict : str = MISSING
  aruco_offset : int = 0
  min_rows : int = 3
  min_points : int = 10



@dataclass 
class AprilConfig:
  _type_: str = "aprilgrid"
  size : Tuple[int, int] = MISSING

  start_id   : int = 0
  tag_family : str = "t36h11"
  tag_length : float = 0.06
  tag_spacing: float = 0.3

  min_rows : int = 2
  min_points : int = 12


@dataclass 
class CheckerboardConfig:
  _type_: str = "checkerboard"
  size : Tuple[int, int] = MISSING
  square_length : float = MISSING



def merge_schema(config, schema):
    merged = OmegaConf.merge(schema, config)
    return struct(**merged)._without('_type_')



def load_config(yaml_file):
  config = OmegaConf.load(yaml_file)
  aruco_params = config.get('aruco_params', {})
  
  boards = {k:OmegaConf.merge(config.common, board) for k, board in config.boards.items()} if 'common' in config\
    else config.boards

  def instantiate_board(config):
    if config._type_ == "charuco":
      schema = OmegaConf.structured(CharucoConfig)
      return CharucoBoard(aruco_params=aruco_params, **merge_schema(config, schema))
    elif config._type_ == "aprilgrid":
      schema = OmegaConf.structured(AprilConfig)
      return AprilGrid(**merge_schema(config, schema))
    else:
      assert False, f"unknown board type: {config._type_}, options are (charuco | aprilgrid | checkerboard)"

  return {k:instantiate_board(board) for k, board in boards.items()}


