import json
import numpy as np

from structs.struct import struct, to_structs
from multical.transform import matrix

from multical.camera import Camera


def transpose_lists(lists):
    return list(zip(*lists))


def import_camera(camera_data):
    intrinsic = camera_data.K
    (w, h) = camera_data.image_size

    return Camera(
        image_size = (w, h),
        intrinsic = np.array(intrinsic),
        dist = np.array(camera_data.dist),
        model = camera_data.get('model', 'standard')
    )


def import_rt(transform_data):
  return matrix.join(np.array(transform_data.R), np.array(transform_data.T))  


def propagate_poses(next, relative):
  poses = {}
  while(len(next) > 0):
    k, t = next.pop(0)

    poses[k] = t
    for r in relative.get(k, []):
      t_dest = t @ r.transform 
      if r.dest in poses:
        assert np.allclose(t_dest, poses[r.dest]), f"inconsistent poses for {r.dest} found"
      else:
        next.append((r.dest, t_dest))

  return poses


def import_pose_graph(poses, names):

  known = {}
  relative = {}

  def check_camera(k):
    assert k in names, f"{k} not found in ({str(names)}"

  def insert_relation(k, t):
    check_camera(k)
    r = relative.get(k, [])
    relative[k] = r + [t]

  for k, v in poses.items():
    t = import_rt(v)

    if "_to_" in k:
      source, dest = k.split("_to_")

      insert_relation(source, struct(dest=dest, transform=np.linalg.inv(t)))
      insert_relation(dest, struct(dest=source, transform=t))
    else:
      check_camera(k)
      known[k] = t

  poses = propagate_poses(list(known.items()), relative)
  missing = set(names) - poses.keys()

  assert len(missing) == 0, f"import_rig: missing camera poses for {str(missing)}"
  return poses



def load_json(filename):
    with open(filename) as json_file:
        d = json.loads(json_file.read())
        return to_structs(d)


def import_cameras(calib_data):
    assert 'cameras' in calib_data, "import_cameras: no camera information in data"
    cameras = {k:import_camera(camera) for k, camera in calib_data.cameras.items()}

    transforms = import_pose_graph(calib_data.camera_poses, cameras)\
      if 'camera_poses' in calib_data else None
    return struct(cameras=cameras, camera_poses=transforms)


def load_calibration(filename):
  calib_data = load_json(filename)
  return import_cameras(calib_data)