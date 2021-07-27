
from operator import add
from pprint import pformat
from typing import Any, Dict, Generic, List, TypeVar
import numpy as np
from structs.numpy import map_arrays, reduce_arrays, shape

from cached_property import cached_property
from scipy.sparse import lil_matrix
from structs.struct import subset
from numbers import Number


class Copyable(object):
  def __init__(self, attrs):
    self._attrs = attrs

  def __getstate__(self):
    return subset(self.__dict__, self._attrs)

  def copy(self, **k):
    """Copy object and change some attribute (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return self.__class__(**d)


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

T = TypeVar('T')

class ParamList(Parameters, Generic[T]):
  def __init__(self, param_objects : List[Parameters], names : List [str] = None):
    self.param_objects = param_objects
    self.names = names

  def __getitem__(self, index):
    if not isinstance(index, Number) and self.names is not None:
      index = self.names.index(index)
    return self.param_objects[index]  

  def __iter__(self):
    return self.param_objects.__iter__()

  def __len__(self):
    return self.param_objects.__len__()

  def __repr__(self):
    if self.names is None:
      return "ParamList " + pformat(self.param_objects)
    else:
      d = {k:obj for k, obj in zip(self.names, self.param_objects)}
      return "ParamList " + pformat(d)

  @cached_property      
  def params(self):
    return [p.param_vec for p in self.param_objects]

  def with_params(self, params):

    updated = [obj.with_param_vec(p) 
      for obj, p in zip(self.param_objects, params)]
    return ParamList(updated, self.names)


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


def build_sparse(params, mapper):
  """ Build a scipy sparse matrix based on pairs of parameter counts and given point indexes 
  """

  total_params = sum([n for n, _ in params])
  sparsity = lil_matrix((mapper.mask_coords.size, total_params), dtype='int16')

  param_count = 0
  for (num_params, point_indexes) in params:
    if point_indexes is not None:
      param_indexes = param_count + np.arange(num_params)
      sparsity[point_indexes.reshape(1, -1), param_indexes.reshape(-1, 1)] = 1

    param_count += num_params

  return sparsity[mapper.mask_coords.ravel()]



class IndexMapper(object):
  """ 
  Small utility to handle mapping parameters to outputs, 
  especially for the construction of the jacobian sparsity matrix.
  """
  def __init__(self, valid_mask):
    self.mask_coords = np.broadcast_to(np.expand_dims(valid_mask, -1), [*valid_mask.shape, 2]) 
    self.indices = np.arange(self.mask_coords.size).reshape(*self.mask_coords.shape)


  def point_indexes(self, i, axis, enabled=True):
    return np.take(self.indices, i, axis=axis).ravel() if enabled else None

  def param_indexes(self, params, axis):
    return [(p.size, self.point_indexes(i, axis=axis))
      for i, p in enumerate(params)]

  def pose_mapping(self, poses, axis, param_size):
    return [(param_size, self.point_indexes(i, axis, optimized))
      for i, optimized in enumerate(poses.valid)]

  def all_points(self, param_size):
    return [(param_size, self.indices)]
