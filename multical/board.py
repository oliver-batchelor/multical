from pprint import pformat, pprint
from typing import Tuple
from cached_property import cached_property
import cv2
import numpy as np

from omegaconf.errors import ValidationError
from omegaconf.omegaconf import MISSING


from structs.struct import pluck, struct, split_dict, choose, subset
from .transform import rtvec
from .optimization.parameters import Parameters

from omegaconf import OmegaConf
from dataclasses import dataclass



def aruco_config(attrs):
  config = cv2.aruco.DetectorParameters_create()
  for k, v in attrs.items():
    assert hasattr(config, k), f"aruco_config: no such detector parameter {k}"
    setattr(config, k, v)  
  return config

empty_detection = struct(corners=np.zeros([0, 2]), ids=np.zeros(0, dtype=np.int))
empty_matches = struct(points1=[], points2=[], ids=[], object_points=[])


def create_dict(name, offset):
  dict_id = getattr(cv2.aruco, f'DICT_{name}')
  aruco_dict=cv2.aruco.getPredefinedDictionary(dict_id)
  aruco_dict.bytesList=aruco_dict.bytesList[offset:]
  return aruco_dict


class CharucoBoard(Parameters):
  def __init__(self, size, square_length, marker_length, min_rows=3, min_points=20, 
    adjusted_points=None, aruco_params=None, aruco_dict='4X4_100', aruco_offset=0):
    
    self.aruco_dict = aruco_dict
    self.aruco_offset = aruco_offset 

    self.size = tuple(size)

    self.marker_length = marker_length
    self.square_length = square_length

    self.adjusted_points = choose(adjusted_points, self.points) 
 
    self.aruco_params = aruco_params or {}
    self.min_rows = min_rows
    self.min_points = min_points


  @cached_property
  def board(self):
    aruco_dict = create_dict(self.aruco_dict, self.aruco_offset)
    width, height = self.size
    return cv2.aruco.CharucoBoard_create(width, height,
      self.square_length, self.marker_length, aruco_dict)

  @cached_property
  def aruco_config(self):
    return aruco_config(self.aruco_params)  

  def export(self):
    return struct(
      type='charuco',
      aruco_dict=self.aruco_dict,
      aruco_offset=self.aruco_offset,
      size = self.size,
      marker_length = self.marker_length,
      square_length = self.square_length,
      aruco_params = self.aruco_params
    )

  def __eq__(self, other):
    return self.export() == other.export()

  @property
  def points(self):
    return self.board.chessboardCorners
  
  @property
  def num_points(self):
    return len(self.points)

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
    corners, ids, _ = cv2.aruco.detectMarkers(image, 
      self.board.dictionary, parameters=aruco_config(self.aruco_params))     
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
      d = self.__getstate__()
      d.update(k)
      return CharucoBoard(**d)

  def __getstate__(self):
    return subset(self.__dict__, ['size', 'adjusted_points', 'aruco_params', 
      'marker_length', 'square_length', 'min_rows', 'min_points',
      'aruco_dict', 'aruco_offset'
    ])



class AprilGrid(Parameters):
  def __init__(self, size, square_length, marker_length, min_rows=3, min_points=20, 
    adjusted_points=None, aruco_params=None, aruco_dict='4X4_100', aruco_offset=0):
    
    self.aruco_dict = aruco_dict
    self.aruco_offset = aruco_offset 

    self.size = tuple(size)

    self.marker_length = marker_length
    self.square_length = square_length

    self.adjusted_points = choose(adjusted_points, self.points) 
 
    self.aruco_params = aruco_params or {}
    self.min_rows = min_rows
    self.min_points = min_points


  @cached_property
  def board(self):
    aruco_dict = create_dict(self.aruco_dict, self.aruco_offset)
    width, height = self.size
    return cv2.aruco.CharucoBoard_create(width, height,
      self.square_length, self.marker_length, aruco_dict)

  @cached_property
  def aruco_config(self):
    return aruco_config(self.aruco_params)  

  def export(self):
    return struct(
      type='charuco',
      aruco_dict=self.aruco_dict,
      aruco_offset=self.aruco_offset,
      size = self.size,
      marker_length = self.marker_length,
      square_length = self.square_length,
      aruco_params = self.aruco_params
    )

  def __eq__(self, other):
    return self.export() == other.export()

  @property
  def points(self):
    return self.board.chessboardCorners
  
  @property
  def num_points(self):
    return len(self.points)

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
    corners, ids, _ = cv2.aruco.detectMarkers(image, 
      self.board.dictionary, parameters=aruco_config(self.aruco_params))     
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
      d = self.__getstate__()
      d.update(k)
      return CharucoBoard(**d)

  def __getstate__(self):
    return subset(self.__dict__, ['size', 'tag_family', 'tag_length', 'tag_spacing', 'min_rows', 'min_points'])
    

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
class AprilGridConfig:
  _type_: str = "aprilgrid"
  size : Tuple[int, int] = MISSING

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

  schema = OmegaConf.structured(CharucoConfig)
  
  def instantiate_board(config):
    if config._type_ == "charuco":
      return CharucoBoard(aruco_params=aruco_params, **merge_schema(config, schema))
    elif config._type_ == "aprilgrid":
      return AprilGrid(aruco_params=aruco_params, **merge_schema(config, schema))
    else:
      assert False, f"unknown board type: {config._type_}, options are (charuco | aprilgrid | checkerboard)"

  return {k:instantiate_board(board) for k, board in boards.items()}



