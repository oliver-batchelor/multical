import numpy as np
from . import matrix


def split(rtvec):
  assert rtvec.shape[-1] == 9
  return rtvec[..., 0:6], rtvec[..., 6:9]

def join(rvec, tvec):
  assert rvec.shape[-1] == 6 and tvec.shape[-1] == 3
  return np.hstack([rvec, tvec])


def renormalise(rvec):
  assert rvec.shape[-1] == 6, f"got: {rvec.shape}"

  x = rvec[..., 0:3]
  y = rvec[..., 3:6]


  e1 = x / np.linalg.norm(x)
  u = y - (e1 * y).sum(-1, keepdims=True) * e1

  e2 = u / np.linalg.norm(u)
  e3 = np.cross(e1, e2)

  return np.stack([e1, e2, e3], axis=-1)


def truncate(rot):
  s = rot.shape[:-2]
  return rot[..., :2].reshape(*s, 6)

def to_matrix(rtvec):
  rvec, tvec = split(rtvec)
  rotation = renormalise(rvec)
  print(rotation.shape, rvec.shape)

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


