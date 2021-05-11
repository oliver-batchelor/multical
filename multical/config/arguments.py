
from dataclasses import dataclass
from multiprocessing import cpu_count
import os
from typing import Optional, Union

from structs.struct import Struct

from simple_parsing import ArgumentParser, choice
from simple_parsing.helpers import list_field

@dataclass 
class Outputs:
  """Output path and filename options"""
  name: str = 'calibration'         # Filename to save outputs, e.g. calibration.json
  output_path: Optional[str] = None # Path to save outputs, uses image path if unspecified
  master: Optional[str] = None  # Use camera as master when exporting (default use first camera)


@dataclass 
class Inputs:
  """ Input files and paths """
  image_path: str     # Path to search for image folders
  boards : Optional[str] = None # Configuration file (YAML) for calibration boards
  camera_pattern: Optional[str] = None  # Camera apth pattern example "{camera}/extrinsic"
  cameras: list[str] = list_field()     # Explicit camera list
 

@dataclass 
class Camera:
  """ Camera model settings """

  fix_aspect: bool = False  # Fix aspect ratio of cameras
  allow_skew: bool = False  # Allow skew parameter in camera intrinsics
  distortion_model: str = choice("standard", "rational", "thin_prism", "tilted", default="standard")
  limit_intrinsic: Optional[int] = 50   # Limit intrinsic images to enable faster initialisation

@dataclass 
class Runtime:
  """ Miscellaneous runtime parameters """
  num_threads: int = cpu_count()  # Number of cpu threads to use
  log_level: str = choice('INFO', 'DEBUG', 'WARN', default='INFO') # Minimum log level
  no_cache: bool = False # Don't attempt to load detections from cache


@dataclass 
class Parameters:
  """ Options for different parameter optimization to enable/disable """ 
  fix_intrinsic: bool = False  # Constant camera intrinsic parameters
  fix_camera_poses: bool = False  # Constant camera pose (extrinsic) parameters 
  fix_board_poses: bool = False  # Constant poses between boards   
  fix_motion: bool = False  # Constant camera motion estimates

  adjust_board: bool = False  # Enable optimization for board non-planarity
  motion_model: bool = choice("rolling", "static", default="static")  # Camera motion model to use
  
 
@dataclass 
class Optimizer:
  """ General optimizer settings including outlier rejection settings """
  
  iter : int = 3 # Iterations of bundle adjustment/outlier rejection
  loss : str = choice('linear', 'soft_l1l', 'huber', 'arctan', default='linear') # Loss function to use in bundle adjustment
  outlier_threshold : float = 5.0 # Threshold for outliers (factor of upper quartile of reprojection error)
  auto_scale : Optional[float] = None # Threshold for auto_scale to reduce outlier influence (factor of upper quartile of reprojection error) - requires non-linear loss


def parse_with(command_type):
    parser = ArgumentParser(prog='multical')
    parser.add_arguments(command_type)

    program = parser.parse_args()
    return program.execute()

