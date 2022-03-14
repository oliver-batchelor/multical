from functools import partial, reduce
import operator
from cached_property import cached_property
import numpy as np
from structs.numpy import shape

from structs.struct import subset, transpose_structs, transpose_lists

from pprint import pformat


from . import camera
from .transform import rtvec, matrix

from structs.struct import struct
from .optimization.parameters import Parameters

from multiprocessing.pool import ThreadPool
from multical.threading import cpu_count

import cv2
from tqdm import tqdm

from structs.struct import split_list



class CameraFisheye(Parameters):
  def __init__(self, image_size, intrinsic, dist, model='standard', fix_aspect=False, has_skew=False):

    assert model in CameraFisheye.model,\
        f"unknown camera model {model} options are {list(self.model.keys())}"

    self.model = model

    self.image_size = tuple(image_size)
    self.intrinsic = intrinsic
    self.dist = np.zeros(5) if dist is None else dist
    self.fix_aspect = fix_aspect
    self.has_skew = has_skew

    # make params accessible through camera model?
  model = struct(
      standard=0,
      fix_k1=cv2.fisheye.CALIB_FIX_K1,
      fix_k2=cv2.fisheye.CALIB_FIX_K2,
      fix_k3=cv2.fisheye.CALIB_FIX_K3,
      fix_k4=cv2.fisheye.CALIB_FIX_K4
  )

  def __str__(self):
    d = dict(intrinsic=self.intrinsic, dist=self.dist,
             image_size=self.image_size)
    return "CameraFisheye " + pformat(d)

  def __repr__(self):
    return self.__str__()

  def approx_eq(self, other):
    assert isinstance(other, CameraFisheye)
    return self.image_size == other.image_size \
        and np.allclose(other.intrinsic, self.intrinsic) \
        and np.allclose(other.dist, self.dist)


  @staticmethod
  def flags():
    return cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

  @staticmethod
  def calibrate(boards, detections, image_size, max_iter=10, eps=1e-3,
                model='standard', fix_aspect=False, has_skew=False, flags=0, max_images=None):
    points = camera.calibration_points(boards, detections)
    if max_images is not None:
      points = camera.top_detection_coverage(points, max_images, image_size)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    flags = CameraFisheye.flags() | flags
    # object points + image points have to get reshaped (adding a dimension)
    objpoints = []
    for obp in points.object_points:
        obp = obp[np.newaxis]
        objpoints.append(obp)
    corners = []
    for cor in points.corners:
        cor = cor[np.newaxis]
        corners.append(cor)
    err, K, dist, _, _ = cv2.fisheye.calibrate(objpoints,
            corners, image_size, None, None, criteria=criteria, flags=flags)

    return CameraFisheye(intrinsic=K, dist=dist, image_size=image_size,
                  model=model, fix_aspect=fix_aspect, has_skew=has_skew), err

  def scale_image(self, factor):
    intrinsic = self.intrinsic.copy()
    intrinsic[:2] *= factor

    return self.copy(intrinsic=intrinsic)

  @cached_property
  def undistort_map(self):
    m, _ = cv2.fisheye.initUndistortRectifyMap(self.intrinsic, self.dist, None,
                                       self.intrinsic, self.image_size, cv2.CV_32FC2)
    return m

  def undistort_points(self, points):
    undistorted = cv2.fisheye.undistortPoints(
        points.reshape(-1, 1, 2), self.intrinsic, self.dist, P=self.intrinsic)
    return undistorted.reshape(*points.shape[:-1], 2)

  def project(self, points):

    projected, _ = cv2.fisheye.projectPoints(
        cv2.UMat(points.reshape(-1, 1, 3)), np.zeros(3), np.zeros(3), self.intrinsic, self.dist)
    return projected.get().reshape(*points.shape[:-1], 2)

  @cached_property
  def focal_length(self):
    fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
    return np.array([fx, fy])

  @cached_property
  def principle_point(self):
    return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]])

  @cached_property
  def skew(self):
    return self.intrinsic[0, 1] if self.has_skew else 0.0


  @cached_property
  def params(self):
    f = self.focal_length
    if self.fix_aspect:
      f = np.array([f.mean(), f.mean()])

    return struct(
        focal_length=f,
        principle_point=self.principle_point,
        skew = np.array([self.skew]),
        dist=self.dist
    )

  def with_params(self, params):

    f = params.focal_length
    fx, fy = f if not self.fix_aspect else (f[0], f[0])

    px, py = params.principle_point
    skew, = params.skew

    intrinsic = [
        [fx,  skew,   px],
        [0,   fy,  py],
        [0,   0,   1],
    ]

    return self.copy(intrinsic=np.array(intrinsic), dist=params.dist)

  def __getstate__(self):
    return subset(self.__dict__,
      ['image_size', 'intrinsic', 'dist', 'fix_aspect', 'has_skew', 'model']
    )

  def copy(self, **k):
    d = self.__getstate__()
    d.update(k)
    return CameraFisheye(**d)


def calibrate_cameras_fisheye(boards, points, image_sizes, **kwargs):

  with ThreadPool() as pool:
    f = partial(CameraFisheye.calibrate, boards, **kwargs)
    return transpose_lists(pool.starmap(f, zip(points, image_sizes)))


def stereo_calibrate_fisheye(cameras, matches, max_iter=60, eps=1e-6,
                     fix_aspect=False, fix_intrinsic=True):

  left, right = cameras
  criteria = (cv2.TERM_CRITERIA_EPS +
              cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

  assert left.image_size == right.image_size
  assert left.model == right.model

  model = left.model
  image_size = left.image_size

  flags = (CameraFisheye.flags(model, fix_aspect) | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
           cv2.fisheye.CALIB_FIX_INTRINSIC * fix_intrinsic)

  err, K1, d1, K2, d2, R, T, E, F = cv2.fisheye.stereoCalibrate(
      matches.object_points, matches.points1, matches.points2,
      left.intrinsic, left.dist,
      right.intrinsic, right.dist,
      image_size, criteria=criteria, flags=flags)

  left = CameraFisheye(dist=d1, intrinsic=K1, image_size=image_size, model=model)
  right = CameraFisheye(dist=d2, intrinsic=K2, image_size=image_size, model=model)

  return left, right, matrix.join(R, T.flatten()), err