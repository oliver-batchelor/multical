import numpy as np
from hypothesis import given, example, assume
from hypothesis.strategies import composite
import hypothesis.strategies as st

import math
from scipy.spatial.transform import Rotation as R
from . import matrix
from . import smooth_6d

def finite_float(**kwargs):
   return st.floats(allow_nan=False, allow_infinity=False, **kwargs)

def uniform_float(min_value=-1.0, max_value=1.0):
  return finite_float(min_value=min_value, max_value=max_value)

def positive_float(**kwargs):
  return finite_float(min_value=0, exclude_min=True, **kwargs)

@composite
def fixed_vector(draw, size, element=uniform_float(), magnitude=finite_float()):

  v = draw(st.lists(element, min_size=size, max_size=size))
  m = draw(magnitude)

  return np.array(v) * m


@composite
def unit_vector(draw, size):
  e = draw(fixed_vector(size, magnitude=uniform_float()))
  
  v = np.array(e)
  n = np.linalg.norm(v)
  assume(n > 0)
  return v / n


angle = lambda: st.floats(min_value=0, max_value=math.pi)
rotation = lambda: unit_vector(size = 4).map(R.from_quat)
rotation_matrix = lambda: unit_vector(size = 4).map(R.from_quat).map(R.as_matrix)


@composite
def transform_matrix(draw, max_translation=1.0):
  t = draw(fixed_vector(size=3, 
    magnitude=uniform_float(-max_translation, max_translation)))

  r = draw(rotation())
  return matrix.join(r.as_matrix(), t)


@given(rotation_matrix(), positive_float(width=32))
def truncate_normalise(r, scale):
  v = smooth_6d.truncate(r) * scale

  m = smooth_6d.renormalise(v)
  assert np.allclose(r, m), f"m = {m}"


if __name__ == '__main__':
  truncate_normalise()