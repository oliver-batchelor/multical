
from operator import add
import numpy as np
from structs.numpy import map_arrays, reduce_arrays

from cached_property import cached_property
from scipy.sparse import lil_matrix


class Parameters(object):

  @cached_property
  def params(self):
    """Return parameters or structs and lists of parameters (as numpy arrays),
    parameters returned will be flattened and optimized and passed to 'with_params'
    """
    raise NotImplementedError()

  def with_params(self, params):
    """ Return a new instance with the provided set of parameters,
    parameters provided must have the same shape as those returned 
    by the 'params' property.
    """
    raise NotImplementedError()

  @cached_property
  def param_vec(self):
    return join(self.params)

  def with_param_vec(self, param_vec):
    updated = split(param_vec, self.params)
    return self.with_params(updated)



def count(params):
  return reduce_arrays(params, np.size, add, 0) 


def split(param_vec, params):
  total = count(params) 
  assert param_vec.size == total,\
     f"inconsistent parameter sizes, got {param_vec.size}, expected {total}"

  def take(arr):
    nonlocal param_vec
    param_vec, params = param_vec[arr.size:], param_vec[:arr.size]
    return params.reshape(arr.shape)

  return map_arrays(params, take)

def join(params):
  params_list = reduce_arrays(params, lambda x: [x], add, []) 
  return np.concatenate([param.ravel() for param in params_list])


def build_sparse(params, valid_mask):
  """ Build a scipy sparse matrix based on pairs of parameter counts and given point indexes 
  """

  total_params = sum([n for n, _ in params])
  sparsity = lil_matrix((valid_mask.size, total_params), dtype='int16')

  param_count = 0
  for (num_params, point_indexes) in params:
    if point_indexes is not None:
      param_indexes = param_count + np.arange(num_params)
      sparsity[point_indexes.reshape(1, -1), param_indexes.reshape(-1, 1)] = 1

    param_count += num_params

  return sparsity[valid_mask.ravel()]
