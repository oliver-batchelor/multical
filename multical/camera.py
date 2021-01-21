from functools import reduce
import operator
from cached_property import cached_property
import numpy as np
import cv2

from structs.struct import transpose_structs, transpose_lists
from structs.numpy import shape

from pprint import pformat

from .transform import rtvec, matrix

from structs.struct import struct
from .optimization.parameters import Parameters

def attributes(obj, keys):
  return {k:getattr(obj, k) for k in keys}


def calibration_points(boards, detections):
  board_detections = transpose_lists(detections)
  points = [board_points(board, detections) for board, detections 
    in zip(boards, board_detections)]

  return reduce(operator.add, points)

def board_points(board, detections):
  non_empty = [d for d in detections if board.has_min_detections(d)]
  assert len(non_empty) > 0, "calibration_points: no points detected"

  detections = transpose_structs(non_empty)
  return detections._extend(
    object_points = [board.points[ids] for ids in detections.ids]
  )


def stereo_calibrate(cameras, matches, max_iter=60, eps=1e-6,
     fix_aspect=False, fix_intrinsic=True):

    left, right = cameras
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)    

    assert left.image_size == right.image_size
    assert left.model == right.model

    model = left.model
    image_size = left.image_size

    flags = (Camera.flags(model, fix_aspect) | cv2.CALIB_USE_INTRINSIC_GUESS |
      cv2.CALIB_FIX_INTRINSIC * fix_intrinsic)

    err, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        matches.object_points, matches.points1, matches.points2,
        left.intrinsic, left.dist, 
        right.intrinsic, right.dist, 
        image_size, criteria=criteria, flags= flags)

    left = Camera(dist = d1, intrinsic = K1, image_size = image_size, model=model)
    right = Camera(dist = d2, intrinsic = K2, image_size = image_size, model=model)

    return left, right, matrix.join(R, T.flatten()), err


class Camera(Parameters):
    def __init__(self, image_size, intrinsic, dist, model='standard', fix_aspect=False):
        
        assert model in Camera.model,\
          f"unknown camera model {model} options are {list(self.model.keys())}"

        self.model = model

        self.image_size = tuple(image_size)
        self.intrinsic = intrinsic
        self.dist = np.zeros(5) if dist is None else dist
        self.fix_aspect = fix_aspect


    model = struct(
      standard = 0,
      rational = cv2.CALIB_RATIONAL_MODEL,
      tilted = cv2.CALIB_TILTED_MODEL,
      thin_prism = cv2.CALIB_THIN_PRISM_MODEL
    )

    def __str__(self):
        d = attributes(self, ['intrinsic', 'dist', 'image_size'])
        return "Camera " + pformat(d)

    def __repr__(self):
        return self.__str__()      


    def approx_eq(self, other):
        assert isinstance(other, Camera)
        return self.image_size == other.image_size \
            and np.allclose(other.intrinsic, self.intrinsic) \
            and np.allclose(other.dist, self.dist)

    @staticmethod
    def flags(model, fix_aspect=False):
      return Camera.model[model] | cv2.CALIB_FIX_ASPECT_RATIO * fix_aspect


    @staticmethod
    def calibrate(boards, detections, image_size, max_iter=10, eps=1e-3, 
        model='standard', fix_aspect=False, flags=0):

      points = calibration_points(boards, detections)
      
      # termination criteria
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
      flags = Camera.flags(model, fix_aspect) | flags
      
      err, K, dist, _, _ = cv2.calibrateCamera(points.object_points, 
          points.corners, image_size, None, None, criteria=criteria, flags=flags)

      return Camera(intrinsic=K, dist=dist, image_size=image_size, 
        model=model, fix_aspect=fix_aspect), err

    def scale_image(self, factor):
      intrinsic = self.intrinsic.copy()
      intrinsic[:2] *= factor

      return self.copy(intrinsic = intrinsic)


    @cached_property
    def undistort_map(self):
      m, _ = cv2.initUndistortRectifyMap(self.intrinsic, self.dist, None, 
        self.intrinsic, self.image_size, cv2.CV_32FC2)
      return m

    def undistort_points(self, points):
      undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), self.intrinsic, self.dist, P=self.intrinsic)
      return undistorted.reshape(*points.shape[:-1], 2)


    def project(self, points, extrinsic=None):
      rvec, tvec = rtvec.split(rtvec.as_rtvec(extrinsic))

      projected, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, tvec, self.intrinsic, self.dist)
      return projected.reshape(*points.shape[:-1], 2)

    @cached_property
    def focal_length(self):
      fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
      return np.array([fx, fy])

    @cached_property
    def principle_point(self):
      return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]])

    @cached_property
    def params(self):
      f = self.focal_length
      if self.fix_aspect:
        f = np.array([f.mean(), f.mean()])

      return struct(
        focal_length = f,
        principle_point = self.principle_point,
        dist = self.dist
      )

    def with_params(self, params):
      f = params.focal_length
      fx, fy = f if not self.fix_aspect else (f[0], f[0])

      px, py = params.principle_point

      intrinsic = [
        [fx,  0,   px],
        [0,   fy,  py],
        [0,   0,   1],
      ]

      return self.copy(intrinsic=np.array(intrinsic), dist=params.dist)


    def copy(self, **k):
        d = dict(image_size = self.image_size, intrinsic = self.intrinsic, 
          dist = self.dist, fix_aspect=self.fix_aspect, model=self.model)
        d.update(k)
        return Camera(**d)
        