
import pickle
from multical.io.logging import info

from structs.struct import struct

def try_load_detections(filename, cache_key={}):
  try:
    with open(filename, "rb") as file:
      loaded = pickle.load(file)
      
      # Check that the detections match the metadata
      if (loaded.get('cache_key', {}) == cache_key):
        info(f"Loaded detections from {filename}")
        return loaded.detected_points
      else:
        info(f"Config changed, not using loaded detections in {filename}")
  except (OSError, IOError, EOFError, AttributeError) as e:
    return None

def write_detections(filename, detected_points, cache_key={}):
  data = struct(
    cache_key = cache_key,
    detected_points = detected_points
  )
  with open(filename, "wb") as file:
    pickle.dump(data, file)