from multical.board.board import Board
from pprint import pformat
from cached_property import cached_property
import cv2
import numpy as np
from .common import *

from structs.struct import struct, choose, subset
from multical.optimization.parameters import Parameters

class CharucoBoard(Parameters, Board):
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

  @cached_property
  def mesh(self):
    return grid_mesh(self.adjusted_points, self.size)


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

    _, corners, ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, self.board)
    
    if ids is None: return empty_detection
    return struct(corners = corners.squeeze(1), ids = ids.squeeze(1))

  def has_min_detections(self, detections):
    return has_min_detections_grid(self.size, detections.ids, 
      min_points=self.min_points, min_rows=self.min_rows)

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




