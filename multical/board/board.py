from cached_property import cached_property
import numpy as np
from structs.struct import struct

class Board(object):
  
  def export(self):
    raise NotImplemented()
  
  def __eq__(self, other):
    raise NotImplemented()
  
  @property
  def points(self) -> np.array:
    raise NotImplemented()
  
  @property
  def num_points(self) -> int:
    raise NotImplemented()
  
  @property 
  def ids(self) -> np.array:
    raise NotImplemented()
  
  @property
  def mesh(self):
    raise NotImplemented()
  
  def draw(self, square_length=50, margin=20):
    raise NotImplemented()
  
  def detect(self, image : np.array) -> struct:    
    raise NotImplemented()
  
  def has_min_detections(self, detections) -> bool:
    raise NotImplemented()

  def estimate_pose_points(self, camera, detections):
    raise NotImplemented()
