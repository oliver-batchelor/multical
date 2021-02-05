import numpy as np
import quaternion

from . import matrix

def lerp(x, y, times):
  t = np.expand_dims(times, times.ndim)
  return x * (1 - t) + y * t

def nlerp(q1, q2, t):
  q = q1 * (1 - t) + q2 * t
  return np.normalized(q)

def interpolate_poses(m1, m2, times):
  r1, t1 = matrix.split(m1)
  r2, t2 = matrix.split(m2)

  t = lerp(t1, t2, times)

  q1 = quaternion.from_rotation_matrix(r1)
  q2 = quaternion.from_rotation_matrix(r2)

  # q = np.slerp_vectorized(q1, q2, times)
  q = nlerp(q1, q2, times)

  r = quaternion.as_rotation_matrix(q)
  return matrix.join(r, t)