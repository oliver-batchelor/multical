import json
from os import path
import numpy as np

from structs.struct import struct, to_dicts
from .transform import matrix

from .optimization.calibration import Calibration

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


def export_extrinsics(camera_names, camera_poses):
  return {k : export_extrinsic(pose, "rig") 
    for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid_poses) 
      if valid}

def export(filename, calib, camera_names, image_names):
  assert isinstance(calib, Calibration)
  
  pose_sets = calib.pose_estimates

  valid_frames = np.flatnonzero(pose_sets.rig.valid_poses)
  rig_poses = pose_sets.rig.poses[valid_frames]

  image_names = np.array(image_names)[valid_frames].tolist()

  data = struct(
    cameras = export_cameras(camera_names, calib.cameras),
    extrinsics = export_extrinsics(camera_names=camera_names, camera_poses=pose_sets.camera),
    rig_poses = [t.tolist() for t in rig_poses],
    boards = [calib.board.export()],

    image_sets = struct(
      rgb = [{camera : path.join(camera, image) for camera in camera_names}
        for image in image_names]
    ),

    board_points = [calib.board.adjusted_points.tolist()]
  )
  

  with open(filename, 'w') as outfile:
      json.dump(to_dicts(data), outfile)