from logging import error
from multical.board.board import Board
from structs.numpy import Table
from multical.board.common import *
from pprint import pformat
from cached_property import cached_property
import cv2
import numpy as np

from structs.struct import struct, choose, subset
from multical.optimization.parameters import Parameters



class AprilGrid(Parameters, Board):
  def __init__(self, size, tag_length, 
      tag_spacing, tag_family='t36h11', border_bits=2, min_rows=2, min_points=12, subpix_region=5, adjusted_points=None):
    
    assert tag_family in self.aruco_dicts
    self.size = tuple(size)

    self.tag_family = tag_family
    self.tag_spacing = tag_spacing
    self.tag_length = tag_length
    self.border_bits = int(border_bits)

    self.adjusted_points = choose(adjusted_points, self.points) 
 
    self.min_rows = min_rows
    self.min_points = min_points
    self.subpix_region = subpix_region

  @cached_property
  def grid(self):
    try:
      import aprilgrid

      w, h = self.size
      family =  getattr(aprilgrid.tagFamilies, self.tag_family)

      return aprilgrid.AprilGrid(h, w, self.tag_length, 
        self.tag_spacing, family=family)
        
    except ImportError as err:     
      error(err)
      error("aprilgrid support depends on apriltags2-ethz, a pip package (linux only)")

  @cached_property
  def board(self):
    spacing_length = self.tag_length * self.tag_spacing
    aruco_dict=cv2.aruco.getPredefinedDictionary(self.aruco_dicts[self.tag_family])

    return cv2.aruco.GridBoard_create(self.size[0], self.size[1], 
      self.tag_length, spacing_length,  aruco_dict)

  def export(self):
    return struct(
      type='aprilgrid',
      tag_family=self.tag_family,
      size = self.size,
      tag_length = self.tag_length,
      tag_spacing = self.tag_spacing,
      border_bits = self.border_bits
    )

  def __eq__(self, other):
    return self.export() == other.export()

  @property
  def points(self):
    tag_ids = range(self.size[0] * self.size[1])
    corners = [self.grid.get_tag_corners_for_id(id) for id in tag_ids]
    points_2d = np.array(corners).reshape(-1, 2)
    return np.concatenate([points_2d, np.zeros([points_2d.shape[0], 1])], axis=1)
      
  @property
  def num_points(self):
    return 4 * self.size[0] * self.size[1]


  @property 
  def tags(self):
    family = getattr(aprilgrid.tagFamilies, self.tag_family)
    return family[:self.size[0] * self.size[1]]

  @property 
  def ids(self):
    return np.arange(self.num_points)

  aruco_dicts = dict(
    t16h5 = cv2.aruco.DICT_APRILTAG_16h5,
    t25h9 = cv2.aruco.DICT_APRILTAG_25h9,
    t36h10 = cv2.aruco.DICT_APRILTAG_36h10, 
    t36h11 = cv2.aruco.DICT_APRILTAG_36h11
  )
  
  def draw(self, square_length=50, margin=20):
    spacing_length = square_length * self.tag_spacing
    dims = [int(square_length * n + spacing_length * (n + 1) + margin * 2) 
      for n in self.size]

    markers = self.board.draw(tuple(dims), marginSize=int(margin + spacing_length), 
      borderBits=int(self.border_bits))

    step = square_length + spacing_length
    for i in range(self.size[0] + 1):
      for j in range(self.size[1] + 1):
        x, y = i * step + margin, j * step + margin
        cv2.rectangle(markers, (int(x), int(y)), (int(x + spacing_length), int(y + spacing_length)),
          (0, 0, 0), cv2.FILLED)

    return markers

  @cached_property
  def mesh(self):
    w, h = self.size
    tag_offsets = np.arange(h * w).reshape(h, w) * 4
    tag_quad = np.arange(0, 4).reshape(1, 4)

    inner_quad = np.array([[
        tag_offsets[0, 1] + 3, tag_offsets[0, 0] + 2, 
        tag_offsets[1, 0] + 1, tag_offsets[1, 1] + 0
      ]])

    tag_quads = tag_quad + tag_offsets.reshape(-1, 1)
    inner_quads = inner_quad + tag_offsets[:h - 1, :w - 1].reshape(-1, 1)

    quads = np.concatenate([tag_quads, inner_quads])

    return struct(
      points=self.adjusted_points, 
      polygons=quad_polygons(quads)
    )


  def __str__(self):
      d = self.export()
      return "AprilGrid " + pformat(d)

  def __repr__(self):
      return self.__str__()      


  def detect(self, image):    
    detections = self.grid.compute_observation(image)

    if not detections.success:
      return empty_detection

    corner_detections = [struct(ids = id * 4 + k % 4, corners=corner)
      for k, id, corner in zip(range(len(detections.ids)), detections.ids, detections.image_points)]

    return subpix_corners(image, Table.stack(corner_detections), self.subpix_region)
          

  def has_min_detections(self, detections):
    tag_ids = detections.ids // 4
    return has_min_detections_grid(self.size, tag_ids, min_points=self.min_points, min_rows=self.min_rows)



  def estimate_pose_points(self, camera, detections):
    return estimate_pose_points(self, camera, detections)


  @cached_property
  def params(self):
    return self.adjusted_points

  def with_params(self, params):
    return self.copy(adjusted_points = params)

  def copy(self, **k):
      d = self.__getstate__()
      d.update(k)
      return AprilGrid(**d)

  def __getstate__(self):
    return subset(self.__dict__, ['size', 'tag_family', 'tag_length', 
      'tag_spacing', 'min_rows', 'min_points', 'border_bits', 
      'subpix_region', 'adjusted_points'])
