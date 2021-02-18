import contextlib
import math
from numbers import Integral
from multical.motion.motion_model import MotionModel
from multical.optimization.pose_set import PoseSet
from multical.board.board import Board
from multical.camera import Camera

from typing import List
from multical.transform.interpolate import interpolate_poses, lerp
import numpy as np

from multical import tables
from multical.transform import matrix, rtvec
# from multical.display import display_pose_projections
from multical.io.logging import LogWriter, info

from . import parameters
from .parameters import ParamList

from structs.numpy import Table, shape
from structs.struct import concat_lists, apply_none, struct, choose, subset, when

from scipy import optimize

from cached_property import cached_property

default_optimize = struct(
  cameras = False,
  boards = False,

  camera_poses = True,
  board_poses = True,
  motion = True
)

def select_threshold(quantile=0.95, factor=1.0):
  def f(reprojection_error):
    return np.quantile(reprojection_error, quantile) * factor
  return f


class Calibration(parameters.Parameters):
  def __init__(self, cameras : ParamList[Camera], boards : ParamList[Board], point_table : Table, 
    camera_poses : PoseSet, board_poses : PoseSet, 
    motion : MotionModel, inlier_mask=None, optimize=default_optimize):

    self.cameras = cameras
    self.boards = boards

    self.point_table = point_table
    self.camera_poses = camera_poses
    self.board_poses = board_poses
    self.motion = motion

    self.optimize = optimize    
    self.inlier_mask = inlier_mask
    
    assert len(self.cameras) == self.size.cameras
    assert camera_poses.size == self.size.cameras
    assert board_poses.size == self.size.boards


  @cached_property 
  def size(self):
    cameras, rig_poses, boards, points = self.point_table._prefix
    return struct(cameras=cameras, rig_poses=rig_poses, boards=boards, points=points)

  @cached_property
  def valid(self):

    valid = (np.expand_dims(self.camera_poses.valid, [1, 2]) & 
      np.expand_dims(self.motion.valid, [0, 2]) &
      np.expand_dims(self.board_poses.valid, [0, 1]))

    return self.point_table.valid & np.expand_dims(valid, valid.ndim)


  @cached_property
  def inliers(self):
    return choose(self.inlier_mask, self.valid)

  @cached_property
  def board_points(self):
    return tables.stack_boards(self.boards)

  @cached_property
  def world_points(self):
    return tables.transform_points(tables.expand_dims(self.board_poses.pose_table, 1), 
      self.board_points)

  @cached_property
  def pose_estimates(self):
    return struct(camera = self.camera_poses.pose_table, 
      board = self.board_poses.pose_table, 
      times = self.motion.frame_poses
    )

  def with_master(self, camera):
    if isinstance(camera, str):
      camera = self.camera_poses.names.index(camera)

    assert isinstance(camera, Integral)
    return self.transform_views(self.camera_poses.poses[camera])


  def transform_views(self, t):
    """ Transform cameras by t^-1 and time poses by t (no change to calibration)
    """ 
    return self.copy(
      camera_poses=self.camera_poses.post_transform(np.linalg.inv(t)), 
      motion = self.motion.pre_transform(t))


  @cached_property
  def projected(self):
    """ Projected points to each image. 
    Returns a table of points corresponding to point_table"""

    return self.motion.project(self.cameras, 
      self.camera_poses.pose_table, self.world_points)


  @cached_property
  def reprojected(self):
    """ Uses the measured points to compute projection motion (if any), 
    to estimate rolling shutter. Only valid for detected points.
    """ 
    return self.motion.project(self.cameras, 
      self.camera_poses.pose_table, self.world_points, self.point_table)



  @cached_property
  def reprojection_error(self):
    return tables.valid_reprojection_error(self.reprojected, self.point_table)

  @cached_property
  def reprojection_inliers(self):
    inlier_table = self.point_table._extend(valid=choose(self.inliers, self.valid))
    return tables.valid_reprojection_error(self.reprojected, inlier_table)


  @cached_property
  def param_objects(self):
    return struct(
      camera_poses = self.camera_poses,
      board_poses = self.board_poses,
      motion = self.motion,

      cameras = self.cameras,
      boards = self.boards
    )

  @cached_property
  def params(self):
    """ Extract parameters as a structs and lists (to be flattened to a vector later)
    """
    all_params =  self.param_objects._map(lambda p: p.param_vec)
    isEnabled = lambda k: self.optimize[k] is True 
    return all_params._filterWithKey(isEnabled)


  def with_params(self, params):
    """ Return a new Calibration object with updated parameters 
    """ 

    updated = {k:self.param_objects[k].with_param_vec(param_vec) 
      for k, param_vec in params.items()}

    return self.copy(**updated)

  @cached_property
  def sparsity_matrix(self):
    """ Sparsity matrix for scipy least_squares,
    Mapping between input parameters and output (point) errors.
    Optional - but optimization runs much faster.
    """
    mapper = parameters.IndexMapper(self.inliers)
    camera_params = self.cameras.param_vec.reshape(self.size.cameras, -1)

    param_mappings = struct(
      camera_poses = self.camera_poses.sparsity(mapper, axis=0),
      board_poses = self.board_poses.sparsity(mapper, axis=2),
      motion = self.motion.sparsity(mapper, axis=1),

      cameras = mapper.param_indexes(camera_params, axis=0),
      boards = concat_lists(
        [mapper.param_indexes(board.param_vec.reshape(-1, 3), axis=3) 
          for board in self.boards])
    )

    mapping_list = [mapping for k, mapping in param_mappings.items() 
      if self.optimize[k] is True]

    return parameters.build_sparse(sum(mapping_list, []), mapper)

  
  def bundle_adjust(self, tolerance=1e-4, f_scale=1.0, max_iterations=100, loss='linear'):
    """ Perform non linear least squares optimization with scipy least_squares
    based on finite differences of the parameters, on point reprojection error
    """

    def evaluate(param_vec):
      calib = self.with_param_vec(param_vec)
      return (calib.reprojected.points - calib.point_table.points)[self.inliers].ravel()

    with contextlib.redirect_stdout(LogWriter.info()):
      res = optimize.least_squares(evaluate, self.param_vec, jac_sparsity=self.sparsity_matrix, 
        verbose=2, x_scale='jac', f_scale=f_scale, ftol=tolerance, max_nfev=max_iterations, method='trf', loss=loss)
  
    return self.with_param_vec(res.x)
  
  def enable(self, **flags):
    for k in flags.keys():
      assert k in self.optimize,\
        f"unknown option {k}, options are {list(self.optimize.keys())}"

    optimize = self.optimize._extend(**flags)
    return self.copy(optimize=optimize)

  def __getstate__(self):
    attrs = ['cameras', 'boards', 'point_table', 'camera_poses', 'board_poses', 
      'motion', 'inlier_mask', 'optimize'
    ]
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy calibration environment and change some attributes (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return Calibration(**d)

  def reject_outliers_quantile(self, quantile=0.95, factor=1.0):
    """ Set inliers based on quantile  """
    threshold = np.quantile(self.reprojection_error, quantile)
    return self.reject_outliers(threshold=threshold * factor)

  
  def reject_outliers(self, threshold):
    """ Set outlier threshold """

    errors, valid = tables.reprojection_error(self.reprojected, self.point_table)
    inliers = (errors < threshold) & valid
    
    num_outliers = valid.sum() - inliers.sum()
    inlier_percent = 100.0 * inliers.sum() / valid.sum()

    info(f"Rejecting {num_outliers} outliers with error > {threshold:.2f} pixels, "
         f"keeping {inliers.sum()} / {valid.sum()} inliers, ({inlier_percent:.2f}%)")

    return self.copy(inlier_mask = inliers)

  def adjust_outliers(self, num_adjustments=4, auto_scale=None, outliers=None, **kwargs):
    info(f"Beginning adjustments ({num_adjustments}) enabled: {self.optimize}, options: {kwargs}")

    for i in range(num_adjustments):
      self.report(f"Adjust_outliers {i}")
      f_scale = apply_none(auto_scale, self.reprojection_error) or 1.0
      if auto_scale is not None:
        info(f"Auto scaling for outliers influence at {f_scale:.2f} pixels")
      
      if outliers is not None:
        self = self.reject_outliers(outliers(self.reprojection_error))

      self = self.bundle_adjust(f_scale=f_scale, **kwargs)
    self.report(f"Adjust_outliers end")
    return self


  def plot_errors(self):
    """ Display plots of error distributions"""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    errors, valid = tables.reprojection_error(self.reprojected, self.point_table)
    errors, valid = errors.ravel(), valid.ravel()

    inliers = self.inliers.ravel()
    outliers = (valid & ~inliers).ravel()
    
    axs[0].scatter(x = np.arange(errors.size)[inliers], y = errors[inliers], marker=".", label='inlier')  
    axs[0].scatter(x = np.arange(errors.size)[outliers], y = errors[outliers], color='r', marker=".", label='outlier')

    axs[1].hist(errors[valid], bins=50, range=(0, np.quantile(errors[valid], 0.999)))

    plt.show()


  def report(self, stage):
    overall = error_stats(self.reprojection_error)
    inliers = error_stats(self.reprojection_inliers)

    if self.inlier_mask is not None:
      info(f"{stage}: reprojection RMS={inliers.rms:.3f} ({overall.rms:.3f}), "
           f"n={inliers.n} ({overall.n}), quantiles={overall.quantiles}")
    else:
      info(f"{stage}: reprojection RMS={overall.rms:.3f}, n={overall.n}, "
           f"quantiles={overall.quantiles}")



def error_stats(errors):  
  mse = np.square(errors).mean()
  quantiles = np.array([np.quantile(errors, n) for n in [0, 0.25, 0.5, 0.75, 1]])
  return struct(mse = mse, rms = np.sqrt(mse), quantiles=quantiles, n = errors.size)




