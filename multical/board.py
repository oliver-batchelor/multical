from pprint import pformat
from typing import Tuple
from cached_property import cached_property
import cv2
import numpy as np

from omegaconf.errors import ValidationError
from omegaconf.omegaconf import MISSING


from structs.struct import pluck, struct, split_dict, choose
from .transform import rtvec
from .optimization.parameters import Parameters

from omegaconf import OmegaConf
from dataclasses import dataclass

def default_aruco_params():
  params = cv2.aruco.DetectorParameters_create()

  # params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
  # params.adaptiveThreshWinSizeMin = 100
  # params.adaptiveThreshWinSizeMax = 700
  # params.adaptiveThreshWinSizeStep = 50
  # params.adaptiveThreshConstant = 0  
  return params

empty_detection = struct(corners=np.zeros([0, 2]), ids=np.zeros(0, dtype=np.int))
empty_matches = struct(points1=[], points2=[], ids=[], object_points=[])


def create_dict(name, offset):
  dict_id = getattr(cv2.aruco, f'DICT_{name}')
  aruco_dict=cv2.aruco.getPredefinedDictionary(dict_id)

  aruco_dict.bytesList=aruco_dict.bytesList[offset:]
  return aruco_dict


class CharucoBoard(Parameters):
  def __init__(self, board, dict_desc, min_rows=3, min_points=20, 
    adjusted_points=None, aruco_params=None):
    
    self.board = board
    self.dict_desc = dict_desc

    self.adjusted_points = choose(adjusted_points, self.points) 
 
      
    self.aruco_params = aruco_params or default_aruco_params()
    self.min_rows = min_rows
    self.min_points = min_points

  @staticmethod
  def create(size, square_length, marker_length, 
    aruco_dict='4X4_100', aruco_offset=0, aruco_params=None, min_rows=3, min_points=20):
      width, height = size
    
      dict_desc = struct(name=aruco_dict, offset=aruco_offset)
      aruco_dict = create_dict(**dict_desc)
      
      board = cv2.aruco.CharucoBoard_create(width, height,
         square_length, marker_length, aruco_dict)

      return CharucoBoard(board, dict_desc, aruco_params=aruco_params, 
        min_rows=min_rows, min_points=min_points)

  def export(self):
    return struct(
      type='charuco',
      dict=self.dict_desc.name,
      offset=self.dict_desc.offset,
      size = self.size,
      marker_length = self.marker_length,
      square_length = self.square_length
    )

  @property
  def points(self):
    return self.board.chessboardCorners
  
  @property
  def num_points(self):
    return len(self.points)

  @property 
  def size(self):
    return self.board.getChessboardSize()

  @property
  def square_length(self):
    return self.board.getSquareLength()

  @property
  def marker_length(self):
    return self.board.getMarkerLength()

  @property 
  def ids(self):
    return np.arange(self.num_points)

  def draw(self, square_length=50, margin=20):
    image_size = [dim * square_length for dim in self.size]
    return self.board.draw(tuple(image_size), marginSize=margin)


  def __str__(self):
      d = self.export()
      return "CharucoBoard " + pformat(d)

  def __repr__(self):
      return self.__str__()      


  def detect(self, image):    
    corners, ids, _ = cv2.aruco.detectMarkers(image, self.board.dictionary, parameters=self.aruco_params)     
    if ids is None: return empty_detection

    _, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, self.board)
    if ids is None: return empty_detection

    return struct(corners = corners.squeeze(1), ids = ids.squeeze(1))


  def has_min_detections(self, detections):
    w, h = self.size
    dims = np.unravel_index(detections.ids, shape=(h, w)) 

    has_rows = [np.unique(d).size >= self.min_rows for d in dims]
    return detections.ids.size >= self.min_points and all(has_rows)

  def estimate_pose_points(self, camera, detections):
      if not self.has_min_detections(detections):
          return None

      undistorted = camera.undistort_points(detections.corners)      
      valid, rvec, tvec = cv2.solvePnP(self.points[detections.ids], 
        undistorted, camera.intrinsic, np.zeros(0))

      if not valid:
        return None

      return rtvec.join(rvec.flatten(), tvec.flatten())

  def refine_points(self, camera, detections, image):
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(detections.corners, detections.ids, image, self.board,
        cameraMatrix=camera.intrinsic, distCoeffs=camera.dist)
    
    if corners is None:
      return empty_detection
    else:
      return struct(corners = corners.squeeze(1), ids = ids.squeeze(1))


  @cached_property
  def params(self):
    return self.adjusted_points

  def with_params(self, params):
    return self.copy(adjusted_points = params)

  def copy(self, **k):
      d = dict(board=self.board, adjusted_points=self.adjusted_points, 
        aruco_params=self.aruco_params, dict_desc=self.dict_desc, 
        min_rows=self.min_rows, min_points=self.min_points)
      d.update(k)
      return CharucoBoard(**d)


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


def load_config(yaml_file):
  config = OmegaConf.load(yaml_file)
  
  boards = {k:OmegaConf.merge(config.common, board) for k, board in config.boards.items()} if 'common' in config\
    else config.boards

  schema = OmegaConf.structured(CharucoConfig)
  board_names, configs = split_dict(boards)

  def instantiate_board(config):
    if config._type_ == "charuco":

      merged = OmegaConf.merge(schema, config)
      merged = struct(**merged)._without('_type_')
      return CharucoBoard.create(**merged)
    else:
      assert False, f"unknown board type: {config._type_}"


  return board_names, [instantiate_board(board) for board in configs]
