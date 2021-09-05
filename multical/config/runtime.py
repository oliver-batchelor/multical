import numpy as np
import multical.image as image
from multical.io.logging import info
from os import path

from structs.struct import struct
from multical.board import load_config, load_calico


def find_board_config(image_path, board_file = None):
  assert board_file is None or path.isfile(board_file), f"board file {board_file} not found"

  board_file = board_file or path.join(image_path, "boards.yaml")
  calico_file = path.join(image_path, "../network_specification_file.txt")
  boards = {}

  if path.isfile(board_file):
    boards = load_config(board_file)
  elif path.isfile(calico_file):
    boards = load_calico(calico_file)      # CALICO board specification file
  else:
    assert False, f"no boards found, use --boards or add boards.yaml to image path"
  
  info("Using boards:")
  for name, b in boards.items():
    info(f"{name} {b}")  
  return boards

def sample_inds(xs, inds):
  return np.array(xs)[inds].tolist()

def find_camera_images(image_path, cameras=None, 
    camera_pattern=None, matching=True, extensions=image.find.image_extensions, limit=None):   
  camera_paths = image.find.find_cameras(image_path, cameras, camera_pattern, extensions=extensions)
  camera_names = list(camera_paths.keys())

  find_images = image.find.find_images_matching if matching else image.find.find_images_unmatched

  image_names, filenames = find_images(camera_paths, extensions=extensions)
  info("Found camera directories {} with {} matching images".format(camera_names, len(image_names)))

  if limit is not None and len(image_names) > limit:
    inds = np.random.permutation(len(image_names))[:limit]
    
    image_names = sample_inds(image_names, inds)
    filenames = [sample_inds(cam_files, inds) for cam_files in filenames]

    info(f"Restricting to {len(inds)} images (--limit_images)")


  return struct(image_path=image_path, cameras=camera_names, image_names=image_names, filenames=filenames)




