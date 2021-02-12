import json
from multical import tables
from os import path
import numpy as np

from structs.struct import struct, to_dicts
from multical.transform import matrix


def export_camera(camera):
  return struct(
      model = camera.model,
      image_size=camera.image_size, 
      K = camera.intrinsic.tolist(),
      dist = camera.dist.tolist()
  )

def export_cameras(camera_names, cameras):
    return {k : export_camera(camera) for k, camera in zip(camera_names, cameras)}

def export_extrinsic(extrinsic, parent):
    r, t = matrix.split(extrinsic)
    return struct (R = r.tolist(), T=t.tolist(), parent=parent)


def export_extrinsics(camera_names, camera_poses, master=None):
  master_name = "rig" if master is None else camera_names[master]

  return {k : export_extrinsic(pose, master_name) 
    for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid) 
      if valid}


def export_poses(pose_table, names):
  return {i:t.poses.tolist() for i, t in zip(names, pose_table._sequence()) 
    if t.valid}


def export(filename, calib, names):  

  board_poses = calib.board_poses.poses
  data = struct(
    cameras = export_cameras(names.camera, calib.cameras),
    extrinsics = export_extrinsics(camera_names=names.camera, 
      camera_poses=tables.inverse(calib.camera_poses)),

    motion = calib.motion.export(),
    board_poses = {k:t.tolist() for k, t in zip(names.board, board_poses)},

    boards = [board.export() for board in calib.boards],

    image_sets = struct(
      rgb = [{camera : path.join(camera, image) for camera in names.camera}
        for image in names.image]
    ),

    board_points = [board.adjusted_points.tolist() for board in calib.boards]
  )
  

  with open(filename, 'w') as outfile:
      json.dump(to_dicts(data), outfile)