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

def export_transform(pose):
    r, t = matrix.split(pose)
    return struct (R = r.tolist(), T=t.tolist())


def export_camera_poses(camera_names, camera_poses):
  return {k : export_transform(pose) 
    for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid) 
      if valid}


def export_relative(camera_names, camera_poses, master):
  assert master in camera_names

  return {k if master is k else f"{k}_to_{master}" : export_transform(pose) 
    for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid) 
      if valid}


def export_poses(pose_table, names=None):
  names = names or [str(i) for i in range(pose_table._size[0])]

  return {i:t.poses.tolist() for i, t in zip(names, pose_table._sequence()) 
    if t.valid}


def export(filename, calib, names, master=None):  
  if master is not None:
    calib = calib.with_master(master)

  camera_poses = calib.camera_poses.pose_table

  data = struct(
    cameras = export_cameras(names.camera, calib.cameras),
    camera_poses = export_camera_poses(names.camera, camera_poses)\
      if master is None else export_relative(names.camera, camera_poses, master),
    
    image_sets = struct(
      rgb = [{camera : path.join(camera, image) for camera in names.camera}
        for image in names.image]
    ),
  )
  
  with open(filename, 'w') as outfile:
    json.dump(to_dicts(data), outfile, indent=2)
