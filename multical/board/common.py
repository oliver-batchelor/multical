import cv2
import numpy as np


from structs.struct import struct
from multical.transform import rtvec


def aruco_config(attrs):
  config = cv2.aruco.DetectorParameters_create()
  for k, v in attrs.items():
    assert hasattr(config, k), f"aruco_config: no such detector parameter {k}"
    setattr(config, k, v)  
  return config

empty_detection = struct(corners=np.zeros([0, 2]), ids=np.zeros(0, dtype=int))
empty_matches = struct(points1=[], points2=[], ids=[], object_points=[])



def create_dict(name, offset):
  dict_id = name if isinstance(name, int)\
    else getattr(cv2.aruco, f'DICT_{name}')

  aruco_dict=cv2.aruco.getPredefinedDictionary(dict_id)
  aruco_dict.bytesList=aruco_dict.bytesList[offset:]
  return aruco_dict


def has_min_detections_grid(grid_size, ids, min_points, min_rows):
  w, h = grid_size
  dims = np.unravel_index(ids, shape=(h, w)) 
  has_rows = [np.unique(d).size >= min_rows for d in dims]
  return ids.size >= min_points and all(has_rows)

def estimate_pose_points(board, camera, detections):
    if not board.has_min_detections(detections):
        return None

    undistorted = camera.undistort_points(detections.corners)      
    valid, rvec, tvec = cv2.solvePnP(board.points[detections.ids], 
      undistorted, camera.intrinsic, np.zeros(0))

    if not valid:
      return None

    return rtvec.join(rvec.flatten(), tvec.flatten())


def subpix_corners(image, detections, window):
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)               
  reshaped = np.array(detections.corners).reshape(-1, 1, 2).astype(np.float32)
  refined = cv2.cornerSubPix(image, reshaped, (window, window), (-1, -1), criteria)
  return detections._extend(corners=refined.reshape(-1, 2))


def quad_polygons(quads):
  assert quads.ndim == 2 and quads.shape[1] == 4

  # Append 4 (number of sides) to each quad
  return np.concatenate([np.full( (quads.shape[0], 1), 4), quads], axis=1)

def grid_mesh(points, size):
  w, h = size
  indices = np.arange(points.shape[0]).reshape(h - 1, w - 1)
  quad = np.array([indices[0, 0], indices[1, 0], indices[1, 1], indices[0, 1]])
  offsets = indices[: h - 2, :w - 2]

  quads = quad.reshape(1, 4) + offsets.reshape(-1, 1)
  return struct(points=points, polygons=quad_polygons(quads))









