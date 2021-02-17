import numpy as np
from structs.numpy import shape
from structs.struct import choose
from . import common

from scipy.spatial.transform import Rotation as R


def homog_points(points):
  padding = np.ones([*points.shape[:-1], 1])
  return  np.concatenate([points, padding], axis=points.ndim - 1)


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


def translation(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, 3]


def rotation(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, :3]


def relative_to(source, dest):
  return  dest @ np.linalg.inv(source)

def error_transform(t, source, dest):
  return np.linalg.norm(t @ source - dest, axis=(1, 2)) 


def center_translation(m):
  r, t = split(m)
  t = t - t.mean(axis=0, keepdims=True)

  x = np.concatenate([t.reshape(-1, 3, 1), r], axis=2)
  return x.transpose(1, 0, 2).reshape(3, -1)

def align_transforms_mean(m1, m2):
  return mean_robust(relative_to(m1, m2))


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
  d = [1, 1, 1 if det > 0 else -1 ]

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

  mask = choose(valid, np.ones(m1.shape[0], dtype=np.bool))

  m = align_transforms_mean(m1[mask], m2[mask])
  errs = error_transform(m, m1, m2)

  inliers = test_outlier(errs, threshold) & mask
  m = align_transforms_mean(m1[inliers], m2[inliers])

  return m, inliers


