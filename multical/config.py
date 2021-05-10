import pickle
import image
from logging import info
from os import path
import pathlib

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

def get_paths(output_path, name="calibration"):

  pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
  
  return struct(
    log_file = path.join(output_path, f"{name}.log"),
    export_file = path.join(output_path, f"{name}.json"),

    detection_cache = path.join(output_path, f"detections.pkl"),
    workspace_file = path.join(output_path, f"{name}.pkl")
  )


def find_camera_images(image_path, cameras=None, camera_pattern=None,  extensions=image.find.image_extensions):   
  camera_paths = image.find.find_cameras(image_path, cameras, camera_pattern, extensions=extensions)
  camera_names = list(camera_paths.keys())

  image_names, filenames = image.find.find_images_matching(camera_paths, extensions=extensions)
  info("Found camera directories {} with {} matching images".format(camera_names, len(image_names)))
  return struct(image_path=image_path, cameras=camera_names, image_names=image_names, filenames=filenames)


def try_load_detections(filename, cache_key={}):
  try:
    with open(filename, "rb") as file:
      loaded = pickle.load(file)
      # Check that the detections match the metadata
      if (loaded.get(cache_key, {}) == cache_key):
        info(f"Loaded detections from {filename}")
        return loaded.detected_points
      else:
        info(f"Config changed, not using loaded detections in {filename}")
  except (OSError, IOError, EOFError, AttributeError) as e:
    return None

def write_detections(self, filename, cache_key={}):
  data = struct(
    cache_key = cache_key,
    detected_points = self.detected_points
  )
  with open(filename, "wb") as file:
    pickle.dump(data, file)
