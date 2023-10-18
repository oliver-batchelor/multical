import numpy as np
import math

from structs.numpy import shape
from structs.struct import choose, struct
from . import common

from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm

def transform(*seq):
  return rtransform(*reversed(seq))

def rtransform(*seq):
  m, *rest = seq
  for t in rest:
    m = m @ t
  return m


def homog_points(points):
  padding = np.ones([*points.shape[:-1], 1])
  return np.concatenate([points, padding], axis=points.ndim - 1)


def transform_homog(t, points):
  points = np.expand_dims(homog_points(points), points.ndim)
  transformed = (t @ points).squeeze(points.ndim - 1)
  return transformed[..., :3]



def join(r, t):
  assert t.ndim == r.ndim - 1 and t.shape[-1] == 3 and r.shape[-2:] == (3, 3)

  d = t.ndim
  m_34 = np.concatenate([r, np.expand_dims(t, d)], axis=d)
  row = np.broadcast_to(np.array([[0, 0, 0, 1]]), (*r.shape[:-2], 1, 4))
  return np.concatenate([m_34, row], axis=d - 1)


def split(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, :3], m[..., :3, 3]

def expand_identity(m, shape=(4, 4)):
    expanded = np.eye(*shape)
    expanded[0:m.shape[0], 0:m.shape[1]] = m
    return expanded


def translation(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, 3]


def rotation(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, :3]


def relative_to(source, dest):
  return dest @ np.linalg.inv(source)


def error_transform(t, source, dest):
  return np.linalg.norm(t @ source - dest, axis=(1, 2))


def center_translation(m):
  r, t = split(m)
  t = t - t.mean(axis=0, keepdims=True)

  x = np.concatenate([t.reshape(-1, 3, 1), r], axis=2)
  return x.transpose(1, 0, 2).reshape(3, -1)


def align_transforms_mean(m1, m2):
  return mean_robust(relative_to(m1, m2))


def mean_rotations(rotations, epsilon=1e-4):
  """Calculates a averaged rotation vector of similar angles.
     See (Single Rotation Averaging 5.2):
     https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf
     """

  assert len(rotations.shape) == 3, "expected Nx3x3 shape"
  n = rotations.shape[0]
  assert n >= 1, "expected 1 or more rotation"

  r_est = rotations[0]
  while True:
    logs = [logm(r_est.transpose() @ r) for r in rotations]
    err = np.array(logs).mean(axis=0)
    r_norm = np.linalg.norm(err)
    if r_norm < epsilon:
      return r_est

    r_est = r_est @ expm(err)

def mean_robust_averaging(m):
  r, t = split(m)
  mean_t = t.mean(axis=0)
  mean_r = mean_rotations(r)

  return join(mean_r, mean_t)

def mean_robust(m):
  from . import rtvec

  rtvecs = rtvec.from_matrix(m)
  return rtvec.to_matrix(common.mean_robust(rtvecs))


def align_transforms_ls(m1, m2):
  """ Least squares solution for XA = B for aligning a collection of
  homogeneous transfomation matrices

  Comparing Two Sets of Corresponding Six Degree of Freedom Data - Shah 2011
  """

  x1, x2 = center_translation(m1), center_translation(m2)
  u, s, vh = np.linalg.svd(x1 @ x2.T)

  det = np.linalg.det(vh @ u.T)
  d = [1, 1, 1 if det > 0 else -1]

  r = vh.T @ np.diag(d) @ u.T
  t = translation(m2).mean(axis=0) - r @ translation(m1).mean(axis=0)

  return join(r, t)


def test_outlier(errs, threshold=2.0):
  uq = np.quantile(errs, 0.75)
  return errs < uq * threshold


def align_transforms_robust(m1, m2, valid=None, threshold=1.5):
  """ As align_transforms, with outlier rejection.
    threshold (float): factor of upper quartile to be determined as an outlier.
  """

  mask = choose(valid, np.ones(m1.shape[0], dtype=bool))

  m = align_transforms_mean(m1[mask], m2[mask])
  errs = error_transform(m, m1, m2)

  inliers = test_outlier(errs, threshold) & mask
  m = align_transforms_mean(m1[inliers], m2[inliers])

  return m, inliers


def pose_errors(p1, p2):
  d = p1 @ np.linalg.inv(p2)
  r, t = split(d)

  return struct(  
    translation = np.linalg.norm(t, axis=(1)),
    rotation_deg = R.magnitude(R.from_matrix(r)) * 180.0 / math.pi,
    frobius = np.linalg.norm(p1 - p2, axis=(1, 2))
  )


