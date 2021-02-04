import contextlib
import math
from multical.transform.interpolate import interpolate_poses, lerp
import numpy as np

from multical import tables
from multical.transform import matrix, rtvec
# from multical.display import display_pose_projections
from multical.io.logging import LogWriter, info

from . import parameters

from structs.numpy import Table, shape
from structs.struct import concat_lists, apply_none, struct, choose, subset, when

from scipy import optimize

from cached_property import cached_property

default_optimize = struct(
  intrinsics = False,
  board = False,
  rolling = False
)

def select_threshold(quantile=0.95, factor=1.0):
  def f(reprojection_error):
    return np.quantile(reprojection_error, quantile) * factor
  return f

def stack_boards(boards):
  padded_points = max([board.num_points for board in boards])

  def pad_points(board):
    points = board.adjusted_points.astype(np.float64)
    return struct(
      points=np.pad(points, [(0, padded_points - points.shape[0]), (0, 0)]),
      valid = np.arange(padded_points) < board.num_points
    )
    
  return Table.stack([pad_points(board) for board in boards]) 

def transform_points(pose_table, board_table):

  points = matrix.transform_homog(
    t      = np.expand_dims(pose_table.poses, 3),
    points = np.expand_dims(board_table.points, [0, 1])
  )    


  valid = (np.expand_dims(pose_table.valid, axis=3) & np.expand_dims(board_table.valid, axis=(0, 1)))
  return Table.create(points=points, valid=valid)


class Calibration(parameters.Parameters):
  def __init__(self, cameras, boards, point_table, pose_estimates, motion_estimates=None, inlier_mask=None, 
      optimize=default_optimize):

    self.cameras = cameras
    self.boards = boards

    self.point_table = point_table
    self.pose_estimates = pose_estimates

    self.motion_estimates = motion_estimates

    self.optimize = optimize    
    self.inlier_mask = inlier_mask
    
    assert len(self.cameras) == self.size.cameras
    assert pose_estimates.camera._shape[0] == self.size.cameras
    assert pose_estimates.rig._shape[0] == self.size.rig_poses
    assert pose_estimates.board._shape[0] == self.size.boards

    assert motion_estimates is None\
       or motion_estimates.shape == pose_estimates.rig.poses.shape

   
  @cached_property 
  def size(self):
    cameras, rig_poses, boards, points = self.point_table._prefix
    return struct(cameras=cameras, rig_poses=rig_poses, boards=boards, points=points)

  @cached_property
  def valid(self):
    return tables.valid(self.pose_estimates, self.point_table) 
 
  @cached_property
  def pose_table(self):
    return tables.expand_poses(self.pose_estimates)

  @cached_property
  def inliers(self):
    return choose(self.inlier_mask, self.valid)

  @cached_property
  def stacked_boards(self):
    return stack_boards(self.boards)


  @cached_property
  def transformed_points(self):
    return transform_points(self.pose_table, self.stacked_boards)

  @cached_property
  def times(self):
    image_heights = np.array([camera.image_size[1] for camera in self.cameras])    
    return self.point_table.points[..., 1] / np.expand_dims(image_heights, (1,2,3))

  @cached_property
  def transformed_rolling(self):
    poses = self.pose_estimates
    start_frame = np.expand_dims(poses.rig.poses, (0, 2, 3))
    end_frame = np.expand_dims(self.frame_motion.poses, (0, 2, 3))
    

    frame_poses = interpolate_poses(start_frame, end_frame, self.times)
    view_poses = np.expand_dims(poses.camera.poses, (1, 2, 3)) @ frame_poses  

    board_points = self.stacked_boards
    board_points_t = matrix.transform_homog(t = np.expand_dims(poses.board.poses, 1), points = board_points.points)

    return struct(
      points = matrix.transform_homog(t = view_poses, points = np.expand_dims(board_points_t, (0, 1))),
      valid = self.valid
    )

  @cached_property
  def transformed_rolling_linear(self):
    poses_start = self.pose_estimates
    poses_end = poses_start._extend(rig=self.frame_motion)

    table_start = tables.expand_poses(poses_start)
    table_end = tables.expand_poses(poses_end)

    start_frame = transform_points(table_start, self.stacked_boards)
    end_frame = transform_points(table_end, self.stacked_boards)

    return struct(
      points = lerp(start_frame.points, end_frame.points, self.times),
      valid = start_frame.valid & end_frame.valid
    )


  @cached_property
  def projected(self):
    """ Projected points from multiplying out poses and then projecting to each image. 
    Returns a table of points corresponding to point_table"""

    transformed = self.transformed_rolling_linear if self.optimize.rolling\
      else self.transformed_points

    image_points = [camera.project(p) for camera, p in 
      zip(self.cameras, transformed.points)]

    return Table.create(points=np.stack(image_points), valid=transformed.valid)

  

  @cached_property
  def reprojection_error(self):
    return tables.valid_reprojection_error(self.projected, self.point_table)

  @cached_property
  def reprojection_inliers(self):
    inlier_table = self.point_table._extend(valid=choose(self.inliers, self.valid))
    return tables.valid_reprojection_error(self.projected, inlier_table)

  @cached_property
  def frame_motion(self):
    rig = self.pose_estimates.rig

    return Table.create(
      poses = self.motion_estimates if self.motion_estimates is not None\
        else rig.poses,
      valid = rig.valid
    )


  @cached_property
  def params(self):
    """ Extract parameters as a structs and lists (to be flattened to a vector later)
    """
    def get_pose_params(poses):
        return rtvec.from_matrix(poses.poses).ravel()

    pose    = self.pose_estimates._map(get_pose_params)

    return struct(
      pose    = struct(camera=pose.camera, rig=pose.rig, board=pose.board),
      camera  = [camera.param_vec for camera in self.cameras
        ] if self.optimize.intrinsics else [], 
      board   = [board.param_vec for board in self.boards
        ] if self.optimize.board else [],

      motion = (get_pose_params(self.frame_motion) 
      if self.optimize.rolling else [])
    )    
  
  def with_params(self, params):
    """ Return a new Calibration object with updated parameters unpacked from given parameter struct
    sets pose_estimates of rig and camera, 
    sets camera intrinsic parameters (if optimized),
    sets adjusted board points (if optimized)
    """
    def update_pose(pose_estimates, pose_params):
      m = rtvec.to_matrix(pose_params.reshape(-1, 6))
      return pose_estimates._update(poses=m)

    pose_estimates = self.pose_estimates._zipWith(update_pose, params.pose)

    cameras = self.cameras
    if self.optimize.intrinsics:
      cameras = [camera.with_param_vec(p) for p, camera in 
        zip(params.camera, self.cameras)]

    boards = self.boards
    if self.optimize.board:
      boards = [board.with_param_vec(board_params) 
        for board, board_params in zip(boards, params.board)]

    motion_estimates = self.motion_estimates
    if self.optimize.rolling:
      motion_estimates = rtvec.to_matrix(params.motion.reshape(-1, 6))

    return self.copy(cameras=cameras, pose_estimates=pose_estimates, 
      boards=boards, motion_estimates=motion_estimates)

  @cached_property
  def sparsity_matrix(self):
    """ Sparsity matrix for scipy least_squares,
    Mapping between input parameters and output (point) errors.
    Optional - but optimization runs much faster.
    """
    inlier_mask = np.broadcast_to(np.expand_dims(self.inliers, -1), [*self.inliers.shape, 2]) 
    indices = np.arange(inlier_mask.size).reshape(*inlier_mask.shape)

    def point_indexes(i, axis, optimized=True):
      return np.take(indices, i, axis=axis).ravel() if optimized else None

    def param_indexes(axis, params):
      return [(p.size, point_indexes(i, axis=axis))
        for i, p in enumerate(params)]

    def pose_mapping(poses, axis):
      return [(6, point_indexes(i, axis, optimized))
        for i, optimized in enumerate(poses.valid)]

    param_mappings = (
      pose_mapping(self.pose_estimates.camera, axis=0) +
      pose_mapping(self.pose_estimates.rig, axis=1) +
      pose_mapping(self.pose_estimates.board, axis=2) +

      param_indexes(0, self.params.camera) +
      concat_lists([param_indexes(3, board.reshape(-1, 3)) 
        for board in self.params.board])
    )

    if self.optimize.rolling:
      param_mappings += pose_mapping(self.frame_motion, axis=1) 

    return parameters.build_sparse(param_mappings, inlier_mask)

  
  def bundle_adjust(self, tolerance=1e-4, f_scale=1.0, max_iterations=100, loss='linear'):
    """ Perform non linear least squares optimization with scipy least_squares
    based on finite differences of the parameters, on point reprojection error
    """

    def evaluate(param_vec):
      calib = self.with_param_vec(param_vec)
      return (calib.projected.points - calib.point_table.points)[self.inliers].ravel()

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
    attrs = ['cameras', 'boards', 'point_table', 'pose_estimates', 
      'motion_estimates', 'inlier_mask', 'optimize'
    ]
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy calibration environment and change some parameters (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return Calibration(**d)

  def reject_outliers_quantile(self, quantile=0.95, factor=1.0):
    """ Set inliers based on quantile  """
    threshold = np.quantile(self.reprojection_error, quantile)
    return self.reject_outliers(threshold=threshold * factor)

  
  def reject_outliers(self, threshold):
    """ Set outlier threshold """

    errors, valid = tables.reprojection_error(self.projected, self.point_table)
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
        info(f"Auto scaling for outliers influence at {f_scale}")
      
      if outliers is not None:
        self = self.reject_outliers(outliers(self.reprojection_error))

      self = self.bundle_adjust(f_scale=f_scale, **kwargs)
    self.report(f"Adjust_outliers end")
    return self


  def plot_errors(self):
    """ Display plots of error distributions"""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    errors, valid = tables.reprojection_error(self.projected, self.point_table)
    errors, valid = errors.ravel(), valid.ravel()

    inliers = self.inliers.ravel()
    outliers = (valid & ~inliers).ravel()
    
    axs[0].scatter(x = np.arange(errors.size)[inliers], y = errors[inliers], marker=".", label='inlier')  
    axs[0].scatter(x = np.arange(errors.size)[outliers], y = errors[outliers], color='r', marker=".", label='outlier')

    axs[1].hist(errors[valid], bins=50, range=(0, np.quantile(errors[valid], 0.999)))

    plt.show()



  # def display(self, images):
  #   """ Display images with poses and reprojections from original detections """
  #   display_pose_projections(self.point_table, self.pose_table, self.board,
  #      self.cameras, images, inliers=self.inliers)    


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




