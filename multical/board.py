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

import apriltags_eth 
import aprilgrid


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


def has_min_detections(board, detections):
  w, h = board.size
  dims = np.unravel_index(detections.ids, shape=(h, w)) 

  has_rows = [np.unique(d).size >= board.min_rows for d in dims]
  return detections.ids.size >= board.min_points and all(has_rows)

def estimate_pose_points(board, camera, detections):
    if not board.has_min_detections(detections):
        return None

    undistorted = camera.undistort_points(detections.corners)      
    valid, rvec, tvec = cv2.solvePnP(board.points[detections.ids], 
      undistorted, camera.intrinsic, np.zeros(0))

    if not valid:
      return None

    return rtvec.join(rvec.flatten(), tvec.flatten())




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
    return has_min_detections(self, detections)

  def estimate_pose_points(self, camera, detections):
    return estimate_pose_points(self, camera, detections)


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
  def __init__(self, size, tag_length, 
      tag_spacing, tag_family='t36h11', adjusted_points=None, border_bits=2, min_rows=2, min_points=12):
    
    assert tag_family in self.aruco_dicts
    self.size = tuple(size)

    self.tag_family = tag_family
    self.tag_spacing = tag_spacing
    self.tag_length = tag_length
    self.border_bits = int(border_bits)

    self.adjusted_points = choose(adjusted_points, self.points) 
 
    self.min_rows = min_rows
    self.min_points = min_points


  @cached_property
  def grid(self):
    w, h = self.size
    family =  getattr(aprilgrid.tagFamilies, self.tag_family)

    return aprilgrid.AprilGrid(h, w, self.tag_length, 
      self.tag_spacing, family=family)

  @cached_property
  def board(self):
    spacing_length = self.tag_length * self.tag_spacing
    aruco_dict=cv2.aruco.getPredefinedDictionary(self.aruco_dicts[self.tag_family])

    return cv2.aruco.GridBoard_create(self.size[0], self.size[1], 
      self.tag_length, spacing_length,  aruco_dict)

  def export(self):
    return struct(
      type='aprilgrid',
      tag_family=self.tag_family,
      size = self.size,
      tag_length = self.tag_length,
      tag_spacing = self.tag_spacing,
      border_bits = self.border_bits
    )

  def __eq__(self, other):
    return self.export() == other.export()

  @property
  def points(self):
    tag_ids = range(self.size[0] * self.size[1])
    corners = [self.grid.get_tag_corners_for_id(id) for id in tag_ids]
    return np.array(corners)

  
  @property
  def num_points(self):
    return 4 * self.size[0] * self.size[1]


  @property 
  def tags(self):
    family = getattr(aprilgrid.tagFamilies, self.tag_family)
    return family[:self.size[0] * self.size[1]]

  @property 
  def ids(self):
    return np.arange(self.num_points)

  aruco_dicts = dict(
    t16h5 = cv2.aruco.DICT_APRILTAG_16h5,
    t25h9 = cv2.aruco.DICT_APRILTAG_25h9,
    t36h10 = cv2.aruco.DICT_APRILTAG_36h10, 
    t36h11 = cv2.aruco.DICT_APRILTAG_36h11
  )
  
  def draw(self, square_length=50, margin=20):
    spacing_length = square_length * self.tag_spacing
    dims = [int(square_length * n + spacing_length * (n + 1) + margin * 2) 
      for n in self.size]

    markers = self.board.draw(tuple(dims), marginSize=int(margin + spacing_length), 
      borderBits=int(self.border_bits))

    step = square_length + spacing_length
    for i in range(self.size[0] + 1):
      for j in range(self.size[1] + 1):
        x, y = i * step + margin, j * step + margin
        cv2.rectangle(markers, (int(x), int(y)), (int(x + spacing_length), int(y + spacing_length)),
          (0, 0, 0), cv2.FILLED)

    return markers

  def __str__(self):
      d = self.export()
      return "AprilGrid " + pformat(d)

  def __repr__(self):
      return self.__str__()      


  def detect(self, image):    
    raise NotImplemented()

  def has_min_detections(self, detections):
    return has_min_detections(self, detections)

  def estimate_pose_points(self, camera, detections):
    return estimate_pose_points(self, camera, detections)



  @cached_property
  def params(self):
    return self.adjusted_points

  def with_params(self, params):
    return self.copy(adjusted_points = params)

  def copy(self, **k):
      d = self.__getstate__()
      d.update(k)
      return AprilGrid(**d)

  def __getstate__(self):
    return subset(self.__dict__, ['size', 'tag_family', 'tag_length', 
      'tag_spacing', 'min_rows', 'min_points', 'border_bits'])
    

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



