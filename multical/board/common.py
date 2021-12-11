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

empty_detection = struct(corners=np.zeros([0, 2], dtype=np.float32), ids=np.zeros(0, dtype=np.int))
empty_matches = struct(points1=[], points2=[], ids=[], object_points=[])



  

def masked_detection_quality(image, detection_sets):
  valid = False

  mask = np.zeros(image.shape[:2], np.int8)
  for detections in detection_sets:
    if detections.ids.size > 0:
      hull = cv2.convexHull(detections.corners)
      if hull is not None:
        valid = True
        cv2.fillConvexPoly(mask, hull.astype(np.int), color=255)

  lap = cv2.Laplacian(image, cv2.CV_32F)
  return lap[mask > 0].std() if valid else 0



def detect_blur_fft(image, low_filter=50):
  # grab the dimensions of the image and use the dimensions to
  # derive the center (x, y)-coordinates
  (h, w) = image.shape
  (cX, cY) = (int(w / 2.0), int(h / 2.0))

  # compute the FFT to find the frequency transform, then shift
  # the zero frequency component (i.e., DC component located at
  # the top-left corner) to the center where it will be more
  # easy to analyze
  fft = np.fft.fft2(image)
  fftShift = np.fft.fftshift(fft)

  # zero-out the center of the FFT shift (i.e., remove low
  # frequencies), apply the inverse shift such that the DC
  # component once again becomes the top-left, and then apply
  # the inverse FFT
  fftShift[cY - low_filter:cY + low_filter, cX - low_filter:cX + low_filter] = 0
  fftShift = np.fft.ifftshift(fftShift)
  recon = np.fft.ifft2(fftShift)

  magnitude = 20 * np.log(np.abs(recon))
  mean = np.mean(magnitude)


  return mean



def image_quality(image):
  lap = cv2.Laplacian(image, cv2.CV_32F)
  median, max = np.quantile(np.abs(lap), [0.5, 1.0])
  return max - median


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









