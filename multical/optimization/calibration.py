import math
import numpy as np

from .. import tables
from ..transform import matrix, rtvec
from ..display import display_pose_projections

from . import parameters

from structs.numpy import Table, shape
from structs.struct import concat_lists, struct, choose

from scipy import optimize

from cached_property import cached_property


class Calibration(parameters.Parameters):
  def __init__(self, cameras, board, point_table, pose_estimates, inliers=None, 
      optimize_intrinsics=False, optimize_board=False):

    self.cameras = cameras
    self.board = board

    self.point_table = point_table
    self.pose_estimates = pose_estimates
    self.optimize_intrinsics = optimize_intrinsics
    self.optimize_board = optimize_board
    
    self.inliers = choose(inliers, self.valid_points)

    
    assert len(self.cameras) == self.size.cameras
    assert pose_estimates.camera._shape[0] == self.size.cameras
    assert pose_estimates.rig._shape[0] == self.size.rig_poses

  @staticmethod
  def initialise(cameras, board, point_table, min_corners=5):
      pose_table = tables.make_pose_table(point_table, board, cameras, min_corners=min_corners)
      pose_estimates = tables.initialise_poses(pose_table)

      return Calibration(cameras, board, point_table, pose_estimates)
   
  @cached_property 
  def size(self):
    cameras, rig_poses, points = self.point_table._prefix
    return struct(cameras=cameras, rig_poses=rig_poses, points=points)

  @cached_property
  def valid_points(self):
    return tables.valid_points(self.pose_estimates, self.point_table) 
 
  @cached_property
  def projected(self):
    """ Projected points from multiplying out poses and then projecting to each image. 
    Returns a table of points corresponding to point_table"""
    pose_table = tables.expand_poses(self.pose_estimates)

    board_points = self.board.adjusted_points.astype(np.float64)
    camera_points = matrix.transform_homog(
      t      = np.expand_dims(pose_table.poses, 2),
      points = np.expand_dims(board_points, [0, 1])
    )

    image_points = [camera.project(p) for camera, p in zip(self.cameras, camera_points)]
    valid_poses = np.repeat(np.expand_dims(pose_table.valid_poses, axis=2), self.size.points, axis=2)

    return Table.create(points=np.stack(image_points), valid_points=valid_poses)

  @cached_property
  def reprojection_error(self):
    return tables.valid_reprojection_error(self.projected, self.point_table)

  @cached_property
  def reprojection_inliers(self):
    inlier_table = self.point_table._extend(valid_points=self.inliers)
    return tables.valid_reprojection_error(self.projected, inlier_table)



  @cached_property
  def params(self):
    """ Extract parameters as a structs and lists (to be flattened to a vector later)
    """
    def get_pose_params(poses):
        return rtvec.from_matrix(poses.poses).ravel()

    return struct(
      pose    = struct(
        camera = get_pose_params(self.pose_estimates.camera),
        rig = get_pose_params(self.pose_estimates.rig)
      ),
      camera  = [camera.param_vec for camera in self.cameras] if self.optimize_intrinsics else [], 
      board   = [self.board.param_vec] if self.optimize_board else []
    )    
  
  def with_params(self, params):
    """ Return a new Calibration object with updated parameters unpacked from given parameter struct
    sets pose_estimates of rig and camera, 
    sets camera intrinsic parameters (if enabled),
    sets adjusted board points (if enabled)
    """
    def from_pose_params(pose_params):
      return rtvec.to_matrix(pose_params.reshape(-1, 6))
    
    pose_estimates = struct(
      rig = self.pose_estimates.rig._update(poses=from_pose_params(params.pose.rig)),
      camera = self.pose_estimates.camera._update(poses=from_pose_params(params.pose.camera))
    )

    cameras = self.cameras
    if self.optimize_intrinsics:
      cameras = [camera.with_param_vec(p) for p, camera in 
        zip(params.camera, self.cameras)]

    board = self.board
    if self.optimize_board:
      board = board.with_param_vec(params.board[0])

    return self.copy(cameras=cameras, pose_estimates=pose_estimates, board=board)

  @cached_property
  def sparsity_matrix(self):
    """ Sparsity matrix for scipy least_squares,
    Mapping between input parameters and output (point) errors.
    Optional - but optimization runs much faster.
    """
    inlier_mask = np.broadcast_to(np.expand_dims(self.inliers, -1), [*self.inliers.shape, 2]) 
    indices = np.arange(inlier_mask.size).reshape(*inlier_mask.shape)

    def point_indexes(i, axis, enabled=True):
      return np.take(indices, i, axis=axis).ravel() if enabled else None

    def param_indexes(axis, params):
      return [(p.size, point_indexes(i, axis=axis))
        for i, p in enumerate(params)]

    def pose_mapping(poses, axis):
      return [(6, point_indexes(i, axis, enabled))
        for i, enabled in enumerate(poses.valid_poses)]

    param_mappings = (
      pose_mapping(self.pose_estimates.camera, axis=0) +
      pose_mapping(self.pose_estimates.rig, axis=1) +
      param_indexes(0, self.params.camera) +
      concat_lists([param_indexes(2, board.reshape(-1, 3)) 
        for board in self.params.board])
    )

    return parameters.build_sparse(param_mappings, inlier_mask)

  
  def bundle_adjust(self, tolerance=1e-4, max_iterations=100, loss='linear'):
    """ Perform non linear least squares optimization with scipy least_squares
    based on finite differences of the parameters, on point reprojection error
    """
    def evaluate(param_vec):
      calib = self.with_param_vec(param_vec)
      return (calib.projected.points - calib.point_table.points)[self.inliers].ravel()
      

    res = optimize.least_squares(evaluate, self.param_vec, jac_sparsity=self.sparsity_matrix, 
      verbose=2, x_scale='jac', ftol=tolerance, max_nfev=max_iterations, method='trf', loss=loss)
  
    return self.with_param_vec(res.x)
 
  
  def enable_intrinsics(self, enabled=True):
    return self.copy(optimize_intrinsics=enabled)

  def enable_board(self, enabled=True):
    return self.copy(optimize_board=enabled)    

  def copy(self, **k):
    """Copy calibration environment and change some parameters (no mutation)"""
    d = dict(cameras=self.cameras, board=self.board, point_table=self.point_table, 
      pose_estimates=self.pose_estimates, inliers=self.inliers, 
      optimize_intrinsics=self.optimize_intrinsics, optimize_board=self.optimize_board)

    d.update(k)
    return Calibration(**d)

  def reject_outliers_quantile(self, quantile=0.95):
    """ Set inliers based on quantile  """
    threshold = np.quantile(self.reprojection_error, quantile)
    return self.reject_outliers(threshold=threshold)

  def reject_outliers_median(self, median_factor=2.5):
    """ Set inliers based on factor of the median """
    median = np.quantile(self.reprojection_error, 0.5)
    return self.reject_outliers(threshold=median * median_factor)

  def reject_outliers(self, threshold):
    """ Set outlier threshold """

    errors, valid = tables.reprojection_error(self.projected, self.point_table)
    inliers = (errors < threshold) & valid
    
    num_outliers = valid.sum() - inliers.sum()
    inlier_percent = 100.0 * inliers.sum() / valid.sum()

    print(f"""Rejecting {num_outliers} outliers with error > {threshold:.2f} pixels,
          keeping {inliers.sum()} / {valid.sum()} inliers, ({inlier_percent:.2f}%)""")

    return self.copy(inliers = inliers)

  def adjust_outliers(self, iterations=4, quantile=0.99):
    for i in range(iterations):
      self.report(f"adjust_outliers: iteration-{i}")
      self = self.reject_outliers_quantile(quantile).bundle_adjust()
    
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



  def display(self, images):
    """ Display images with poses and reprojections from original detections """
    pose_table = tables.expand_poses(self.pose_estimates)
    display_pose_projections(self.point_table, pose_table, self.board,
       self.cameras, images, inliers=self.inliers)    


  def report(self, stage):
    def stats(errors):
      mse = np.square(errors).mean()
      return struct(mse = mse, rms = np.sqrt(mse), n = errors.size)

    errors = stats(self.reprojection_error)
    inliers = stats(self.reprojection_inliers)
    
    print(f"{stage}: reprojection RMS={errors.rms} n={errors.n}, inlier RMS={inliers.rms} n={inliers.n}")









