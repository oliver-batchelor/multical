import numpy as np

from structs.struct import transpose_structs, lens
from structs.numpy import struct, Table, shape

from .transform import rtvec, matrix
from . import graph

from scipy.spatial.transform import Rotation as R


def fill_sparse(n, values, ids):
  dense = np.zeros((n, *values.shape[1:]), dtype=values.dtype)
  dense[ids] = values

  mask = np.full(n, False)
  mask[ids] = True
  return dense, mask

def fill_sparse_tile(n, values, ids, tile):
  assert tile.shape == values.shape[1:]
  dense = np.broadcast_to(np.expand_dims(tile, 0), (n, *tile.shape)).copy()
  dense[ids] = values

  mask = np.full(n, False)
  mask[ids] = True
  return dense, mask



def sparse_points(points):
  ids = np.flatnonzero(points.valid_points)
  return struct(corners=points.points[ids], ids=ids)


invalid_pose = struct(poses=np.eye(4), num_points=0, valid_poses=False)
identity_pose = struct(poses=np.eye(4), num_points=0, valid_poses=True)


def extract_pose(points, board, camera, min_corners=20):
  detections = sparse_points(points)
  poses = board.estimate_pose_points(
      camera, detections, min_corners=min_corners)

  return struct(poses=rtvec.to_matrix(poses), num_points=len(detections.ids), valid_poses=True)\
      if poses is not None else invalid_pose


def make_pose_table(point_table, board, cameras, min_corners=20):

  poses = [[extract_pose(points, board, camera, min_corners)
            for points in points_camera._sequence()]
           for points_camera, camera in zip(point_table._sequence(), cameras)]

  return make_2d_table(poses)


def make_point_table(detections, board):
  def extract_points(frame_dets):
    
    points, mask = fill_sparse(
        board.num_points, frame_dets.corners, frame_dets.ids)
    return struct(points=points, valid_points=mask)

  points = [[extract_points(d) for d in cam_dets]
            for cam_dets in detections]

  return make_2d_table(points)


def make_2d_table(items):
  rows = [Table.stack(row) for row in items]
  return Table.stack(rows)


dimensions = struct(
    camera=0,
    frame=1
)


def map_pairs(f, table, axis=0):
  n = table._prefix[axis]
  pairs = {}

  for i in range(n):
    row_i = table._index[i]
    for j in range(i + 1, n):
      pairs[(i, j)] = f(row_i, table._index[j])

  return pairs


def matching_points(points, board, cam1, cam2):
  points1, points2 = points._index[cam1], points._index[cam2]
  matching = []

  for i, j in zip(points1._sequence(0), points2._sequence(0)):
    row1, row2, ids = common_entries(i, j, 'valid_points')
    matching.append(struct(
        points1=row1.points,
        points2=row2.points,
        object_points=board.points[ids],
        ids=ids
    )
    )

  return transpose_structs(matching)


def common_entries(row1, row2, mask_key):
  valid = np.nonzero(row1[mask_key] & row2[mask_key])
  return row1._index[valid], row2._index[valid], valid


def pattern_overlaps(table, axis=0):
  n = table._prefix[axis]
  overlaps = np.zeros([n, n])

  for i in range(n):
    for j in range(i + 1, n):
      row_i, row_j = table._index_select(
          i, axis=axis), table._index_select(j, axis=axis)

      has_pose = (row_i.valid_poses & row_j.valid_poses)
      weight = np.min([row_i.num_points, row_j.num_points], axis=0)
      overlaps[i, j] = overlaps[j, i] = np.sum(
          has_pose.astype(np.float32) * weight)
  return overlaps


def estimate_transform(table, i, j, axis=0):
  poses_i = table._index_select(i, axis=axis).poses
  poses_j = table._index_select(j, axis=axis).poses

  t, errs, inliers = matrix.align_transforms_robust(poses_i, poses_j)

  return t


def estimate_poses(table, axis=0, hop_penalty=0.9):
  n = table._prefix[axis]
  overlaps = pattern_overlaps(table, axis=axis)
  master, pairs = graph.select_pairs(overlaps, hop_penalty)

  poses = {master: np.eye(4)}

  for parent, child in pairs:
    t = estimate_transform(table, parent, child, axis=axis)
    poses[child] = t @ poses[parent]

  valid_ids = sorted(poses)
  pose_table = np.array([poses[k] for k in valid_ids])

  values, mask = fill_sparse_tile(n, pose_table, valid_ids, np.eye(4))
  return Table.create(poses=values, valid_poses=mask), master


def multiply_masked(poses1, poses2):
  return Table.create(
      poses=np.expand_dims(poses1.poses, 1) @ np.expand_dims(poses2.poses, 0),
      valid_poses=np.expand_dims(
          poses1.valid_poses, 1) & np.expand_dims(poses2.valid_poses, 0)
  )


def expand_poses(estimates):
  return multiply_masked(estimates.camera, estimates.rig)


def valid_points(estimates, point_table):
  valid_poses = np.expand_dims(estimates.camera.valid_poses, 1) & np.expand_dims(
      estimates.rig.valid_poses, 0)
  return point_table.valid_points & np.expand_dims(valid_poses, valid_poses.ndim)


def valid_reprojection_error(points1, points2):
  errors, mask = reprojection_error(points1, points2)
  return errors[mask]


def reprojection_error(points1, points2):
  mask = points1.valid_points & points2.valid_points
  error = np.linalg.norm(points1.points - points2.points, axis=-1)
  return error, mask


def relative_between(table1, table2):
  common1, common2, _ = common_entries(table1, table2, mask_key='valid_poses')
  t, errs, inliers = matrix.align_transforms_robust(
      common1.poses, common2.poses)

  return t


def relative_between_inv(table1, table2):
  return np.linalg.inv(relative_between(inverse(table1), inverse(table2)))


def inverse(table):
  return struct(poses=np.linalg.inv(table.poses), valid_poses=table.valid_poses)


def pre_multiply(table, t):
  return struct(poses=table.poses @ t, valid_poses=table.valid_poses)


def stereo_calibrate(points, board, cameras, i, j, **kwargs):
  matching = matching_points(points, board, i, j)
  return stereo_calibrate((cameras[i], cameras[j]), matching, **kwargs)


def initialise_poses(pose_table):

    # Find relative transforms between cameras and rig poses
  camera, cam_master = estimate_poses(pose_table, axis=0)
  rig, rig_master = estimate_poses(pose_table, axis=1)

  # Initialise relative camera poses and rig poses
  relative = struct(camera=camera, rig=rig)


  # Given relative transforms, find the absolute transform t
  # camera @ rig @ t = pose
  t = relative_between_inv(expand_poses(relative), pose_table)
  # t = pose_table._index[cam_master, rig_master].poses

  # adjust rig poses to give original poses back
  return lens.rig.poses.set(relative, rig.poses @ t)
