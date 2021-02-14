import numpy as np
from . import matrix
from scipy.spatial.transform import Rotation as R


def split(qtvec):
  assert qtvec.shape[-1] == 7
  return qtvec[..., 0:4], qtvec[..., 4:7]

def join(q, tvec):
  assert q.shape[-1] == 4 and tvec.shape[-1] == 3
  return np.hstack([q, tvec])

size = 7


def truncate(rot):
  s = rot.shape[:-2]
  return rot[..., :2, :].reshape(*s, 6)

def to_matrix(qtvec):
  q, tvec = split(qtvec)
  rotation = R.from_quat(q).as_matrix()
  return matrix.join(rotation, tvec)

def from_matrix(m):
  rot, t = matrix.split(m)
  q = R.from_matrix(rot).as_quat()
  return join(q, t)


def multiply(qtvecs1, qtvecs2):
  m = to_matrix(qtvecs1) @ to_matrix(qtvecs2)
  return from_matrix(m)


def relative_to(source, dest):
  m = matrix.relative_to(to_matrix(source), to_matrix(dest))
  return from_matrix(m)


