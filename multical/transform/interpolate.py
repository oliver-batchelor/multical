import numpy as np
import quaternion

from . import matrix

def lerp(x, y, times):
  t = np.expand_dims(times, times.ndim)
  return x * (1 - t) + y * t

def interpolate_poses(m1, m2, times):
  r1, t1 = matrix.split(m1)
  r2, t2 = matrix.split(m2)

  t = lerp(t1, t2, times)

  q1 = quaternion.from_rotation_matrix(r1)
  q2 = quaternion.from_rotation_matrix(r2)

  q = np.slerp_vectorized(q1, q2, times)

  r = quaternion.as_rotation_matrix(q)
  return matrix.join(r, t)