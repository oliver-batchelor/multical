from cached_property import cached_property
import cv2
import numpy as np

from structs.struct import struct
from .transform import rtvec
from .optimization.parameters import Parameters

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

  if offset > 0:
    aruco_dict.bytesList=aruco_dict.bytesList[offset:]
  return aruco_dict


class CharucoBoard(Parameters):
  def __init__(self, board, dict_desc, adjusted_points=None, aruco_params=None):
    
    self.board = board
    self.dict_desc = dict_desc

    self.adjusted_points = (adjusted_points 
      if adjusted_points is not None else self.points)
      
    self.aruco_params = aruco_params or default_aruco_params()

  @staticmethod
  def create(size, square_length, marker_length, 
    aruco_dict='4X4_100', aruco_offset=0, aruco_params=None):
      width, height = size
    
      dict_desc = struct(name=aruco_dict, offset=aruco_offset)
      aruco_dict = create_dict(**dict_desc)
      
      board = cv2.aruco.CharucoBoard_create(width, height,
         square_length, marker_length, aruco_dict)

      return CharucoBoard(board, dict_desc, aruco_params=aruco_params)

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

  def draw(self, image_size):
      return self.board.draw(tuple(image_size))


  def detect(self, image):    
    corners, ids, _ = cv2.aruco.detectMarkers(image, self.board.dictionary, parameters=self.aruco_params)     
    if ids is None: return empty_detection

    _, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, self.board)
    if ids is None: return empty_detection

    return struct(corners = corners.squeeze(1), ids = ids.squeeze(1))



  def estimate_pose_points(self, camera, detections, min_corners=20):
      if len(detections.corners) < min_corners:
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
        aruco_params=self.aruco_params, dict_desc=self.dict_desc)
      d.update(k)
      return CharucoBoard(**d)