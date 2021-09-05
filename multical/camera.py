from functools import partial, reduce
import operator
from cached_property import cached_property
import numpy as np
import cv2
from structs.numpy import shape

from structs.struct import subset, transpose_structs, transpose_lists

from pprint import pformat

from .transform import rtvec, matrix

from structs.struct import struct
from .optimization.parameters import Parameters

from multiprocessing.pool import ThreadPool
from multical.threading import cpu_count

import cv2
from tqdm import tqdm

from structs.struct import split_list



class Camera(Parameters):
  def __init__(self, image_size, intrinsic, dist, model='standard', fix_aspect=False, has_skew=False):

    assert model in Camera.model,\
        f"unknown camera model {model} options are {list(self.model.keys())}"

    self.model = model

    self.image_size = tuple(image_size)
    self.intrinsic = intrinsic
    self.dist = np.zeros(5) if dist is None else dist
    self.fix_aspect = fix_aspect
    self.has_skew = has_skew

  model = struct(
      standard=0,
      rational=cv2.CALIB_RATIONAL_MODEL,
      tilted=cv2.CALIB_TILTED_MODEL,
      thin_prism=cv2.CALIB_THIN_PRISM_MODEL
  )

  def __str__(self):
    d = dict(intrinsic=self.intrinsic, dist=self.dist,
             image_size=self.image_size)
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
                model='standard', fix_aspect=False, has_skew=False, flags=0, max_images=None):

    points = calibration_points(boards, detections)
    if max_images is not None:
      points = top_detection_coverage(points, max_images, image_size)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    flags = Camera.flags(model, fix_aspect) | flags

    err, K, dist, _, _ = cv2.calibrateCamera(points.object_points,
            points.corners, image_size, None, None, criteria=criteria, flags=flags)

    return Camera(intrinsic=K, dist=dist, image_size=image_size,
                  model=model, fix_aspect=fix_aspect, has_skew=has_skew), err

  def scale_image(self, factor):
    intrinsic = self.intrinsic.copy()
    intrinsic[:2] *= factor

    return self.copy(intrinsic=intrinsic)

  @cached_property
  def undistort_map(self):
    m, _ = cv2.initUndistortRectifyMap(self.intrinsic, self.dist, None,
                                       self.intrinsic, self.image_size, cv2.CV_32FC2)
    return m

  def undistort_points(self, points):
    undistorted = cv2.undistortPoints(
        points.reshape(-1, 1, 2), self.intrinsic, self.dist, P=self.intrinsic)
    return undistorted.reshape(*points.shape[:-1], 2)

  def project(self, points):

    projected, _ = cv2.projectPoints(
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
    return Camera(**d)


def board_correspondences(board, detections):
  non_empty = [d for d in detections if board.has_min_detections(d)]
  if len(non_empty) == 0:
    return struct(corners = [], object_points=[], ids=[])

  detections = transpose_structs(non_empty)
  return detections._extend(
      object_points=[board.points[ids].astype(np.float32) for ids in detections.ids],
      corners=[corners.astype(np.float32) for corners in detections.corners]
  )


def board_frames(board, detections):
  non_empty = [d for d in detections if board.has_min_detections(d)]
  return len(non_empty)


def index_list(xs, indexes):
  return np.array(xs, dtype=object)[indexes].tolist()


def coverage(corners, bins):
  hist, _, _ = np.histogram2d(corners[:, 0], corners[:, 1], bins)
  counts = np.count_nonzero(hist)

  return counts

def image_bins(image_size, approx_bins=10):
  bin_size = min(image_size[0] / approx_bins, image_size[1] / approx_bins)

  return [np.linspace(0, image_size[axis], int(image_size[axis] / bin_size)) 
    for axis in [0, 1]]


def top_detection_coverage(detections, k, image_size, approx_bins=10, jitter=0.1):
  bins = image_bins(image_size, approx_bins=10)
  bin_jitter = jitter * (approx_bins * approx_bins)

  sizes = [-coverage(corners, bins) + np.random.normal(0, bin_jitter) 
    for corners in detections.corners]

  sorted = detections._map(index_list, np.argsort(sizes))
  return sorted._map(lambda xs: xs[:k])


def calibration_points(boards, detections):

  board_detections = transpose_lists(detections)
  board_points = [board_correspondences(board, detections) for board, detections
                  in zip(boards, board_detections)]

  return reduce(operator.add, board_points)


def calibrate_cameras(boards, points, image_sizes, **kwargs):

  with ThreadPool() as pool:
    f = partial(Camera.calibrate, boards, **kwargs)
    return transpose_lists(pool.starmap(f, zip(points, image_sizes)))


def undistort_image(args):
  image, undistort_map = args
  return cv2.remap(image, undistort_map, None, cv2.INTER_CUBIC)


def undistort_images(images, cameras, j=cpu_count(), chunksize=4):
  with ThreadPool(processes=j) as pool:
    image_pairs = [(image, camera.undistort_map)
                   for camera, cam_images in zip(cameras, images)
                   for image in cam_images]

    loader = pool.imap(undistort_image, image_pairs, chunksize=chunksize)
    undistorted = list(tqdm(loader, total=len(image_pairs)))

    return split_list(undistorted, [len(i) for i in images])


def stereo_calibrate(cameras, matches, max_iter=60, eps=1e-6,
                     fix_aspect=False, fix_intrinsic=True):

  left, right = cameras
  criteria = (cv2.TERM_CRITERIA_EPS +
              cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

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
      image_size, criteria=criteria, flags=flags)

  left = Camera(dist=d1, intrinsic=K1, image_size=image_size, model=model)
  right = Camera(dist=d2, intrinsic=K2, image_size=image_size, model=model)

  return left, right, matrix.join(R, T.flatten()), err
