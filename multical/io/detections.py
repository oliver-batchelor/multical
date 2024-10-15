
import pickle
from multical.io.logging import info
import os
from structs.struct import struct

def try_load_detections(filename, cache_key={}):
  try:
    with open(filename, "rb") as file:
      loaded = pickle.load(file)
      
      # Check that the detections match the metadata
      if (loaded.get('cache_key', {}) == cache_key or check_dataset_similarity(loaded, cache_key)):
        info(f"Loaded detections from {filename}")
        return loaded.detected_points
      else:
        info(f"Config changed, not using loaded detections in {filename}")
  except (OSError, IOError, EOFError, AttributeError) as e:
    return None

def check_dataset_similarity(loaded, cache_key):
  '''
  Checks whether the loaded datasets file format is similar to cached dataset
  '''
  filenames = loaded.cache_key['filenames']
  caches = cache_key['filenames']

  assert len(filenames) == len(caches)
  for i in range(len(filenames)):
    assert len(filenames[i]) == len(caches[i])
    for j in range(len(filenames[i])):
      file_dirs = find_char(filenames[i][j])
      cache_dirs = find_char(caches[i][j])
      if (file_dirs[-3:] == cache_dirs[-3:]):
        continue
      else:
        return False

  return True

def find_char(str):
  '''
  Helper function for check_dataset_similarity function
  '''
  x = str.split("\\")
  y = str.split("/")
  return x if len(x) > len(y) else y

def write_detections(filename, detected_points, cache_key={}):
  data = struct(
    cache_key = cache_key,
    detected_points = detected_points
  )
  with open(filename, "wb") as file:
    pickle.dump(data, file)