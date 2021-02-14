import numpy as np
from . import matrix


def split(rtvec):
  assert rtvec.shape[-1] == 9
  return rtvec[..., 0:6], rtvec[..., 6:9]

def join(rvec, tvec):
  assert rvec.shape[-1] == 6 and tvec.shape[-1] == 3
  return np.hstack([rvec, tvec])

size = 9

def renormalise(rvec):
  def normalise(v):
    return v / np.linalg.norm(v)

  assert rvec.shape[-1] == 6, f"got: {rvec.shape}"
  rvec = rvec.reshape(*rvec.shape[:-1], 2, 3)

  a1 = rvec[..., 0, :]
  a2 = rvec[..., 1, :]

  b1 = normalise(a1)
  b2 = normalise(a2 - (b1 * a2).sum(-1, keepdims=True) * b1)

  b3 = np.cross(b1, b2)
  return np.stack([b1, b2, b3], axis=-2)


def truncate(rot):
  s = rot.shape[:-2]
  return rot[..., :2, :].reshape(*s, 6)

def to_matrix(rtvec):
  rvec, tvec = split(rtvec)
  rotation = renormalise(rvec)

  return matrix.join(rotation, tvec)

def from_matrix(m):
  rot, t = matrix.split(m)
  rvec = truncate(rot)
  return join(rvec, t)


def multiply(rtvecs1, rtvecs2):
  m = to_matrix(rtvecs1) @ to_matrix(rtvecs2)
  return from_matrix(m)


def relative_to(source, dest):
  m = matrix.relative_to(to_matrix(source), to_matrix(dest))
  return from_matrix(m)


