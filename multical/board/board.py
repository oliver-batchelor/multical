from cached_property import cached_property
import numpy as np
from structs.struct import struct

class Board(object):
  
  def export(self):
    raise NotImplementedError()
  
  def __eq__(self, other):
    raise NotImplementedError()
  
  @property
  def points(self) -> np.array:
    raise NotImplementedError()
  
  @property
  def num_points(self) -> int:
    raise NotImplementedError()
  
  @property 
  def ids(self) -> np.array:
    raise NotImplementedError()

  @property
  def size_mm(self):
    raise NotImplementedError()

  
  @property
  def mesh(self):
    raise NotImplementedError()
  
  def draw(self, pixels_mm=1, margin=20):
    raise NotImplementedError()
  
  def detect(self, image : np.array) -> struct:    
    raise NotImplementedError()
  
  def has_min_detections(self, detections) -> bool:
    raise NotImplementedError()

  def estimate_pose_points(self, camera, detections):
    raise NotImplementedError()
