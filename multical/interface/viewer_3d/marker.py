from functools import partial
from multical import tables
from structs.numpy import shape
from structs.struct import choose, map_list, struct
from .vtk_tools import vtk_transform
import numpy as np

from multical.transform import matrix
import pyvista as pv


def view_projection(camera, size=1):

  intrinsic = matrix.join(camera.intrinsic, np.zeros(3))
  proj = np.linalg.inv(intrinsic)

  # pre-multiply by depth because the projected coordinate is a homogeneous coordinate
  # i.e. [x * z, y * z, z]
  # alternatively, we can project the points at the normalized depth (1)
  w, h = camera.image_size[0] * size, camera.image_size[1] * size
  corners = np.array([
      [0, 0, size, 1],
      [w, 0, size, 1],
      [w, h, size, 1],
      [0, h, size, 1],
  ]).transpose()

  return np.transpose(proj @ corners)[:, 0:3]

view_triangles = np.hstack([[3, 0, 2, 1], [3, 0, 3, 2], [3, 0, 4, 3], [3, 0, 1, 4]])

def projection_corners(camera, scale=1):
  return np.concatenate([
      np.zeros((1, 3)),
      view_projection(camera, size=scale)
  ])

def axis_points(scale=1):
  origin = [0, 0, 0]
  corners = np.array([
      origin, [1, 0, 0], origin, [0, 1, 0], origin, [0, 0, 1],
    ]
  )
  return corners * scale

def axis_marker(scale = 1):
  red =   [1, 0, 0]
  green = [0, 1, 0]
  blue =  [0, 0, 1]

  colors = np.array([red, red, green, green, blue, blue])
  lines = np.array([
    [2, 0, 1],
    [2, 2, 3],
    [2, 4, 5]
  ])

  mesh = pv.PolyData(axis_points(scale))
  mesh.lines = lines
  mesh['colors'] = colors * 255
  return mesh


def image_projection(camera, distance=1):
  corners = view_projection(camera, size=distance)
  triangles = np.array([[4, 2, 1, 0, 3]])

  uvs = np.array([(0, 1), (1, 1), (1, 0), (0, 0)])

  mesh = pv.PolyData(corners, triangles)
  mesh.t_coords = uvs
  return mesh
 

def scaled(pose, scale):
  return pose @ np.diag([scale, scale, scale, 1.0]) 

class Marker():
  def __init__(self, viewer, mesh, pose, options, scale=1):
    self.actor = viewer.add_mesh(pv.PolyData(mesh), **options)
    self.showing = True
    self.set_transform(pose, scale)

  def set_transform(self, pose, scale=1):
    self.pose = pose
    self.scale = scale

    transform = scaled(self.pose.poses, self.scale)
    self.actor.SetUserTransform(vtk_transform(transform))
    self.actor.SetVisibility(self.pose.valid and self.showing)

  def set_color(self, color, opacity=1):
    p = self.actor.GetProperty()
    p.SetColor(*color)
    p.SetOpacity(opacity)    

  def set_ambient(self, ambient):
    p = self.actor.GetProperty()
    p.SetAmbient(ambient)

  def set_point_size(self, size):
    p = self.actor.GetProperty()
    p.SetPointSize(size)


  def show(self, shown):
    self.showing = shown
    self.actor.SetVisibility(self.pose.valid and self.showing)

   
class SceneMeshes():
  def __init__(self, calib, camera_scale=1):
    self.board =  [pv.PolyData(board.mesh.points, board.mesh.polygons)
       for board in calib.boards]

    self.camera_scale = camera_scale

    self.camera_projections = [projection_corners(camera, scale=1) for camera in calib.cameras]
    self.camera = [pv.PolyData(self.camera_scale * proj, view_triangles) 
      for proj in self.camera_projections]
     
    self.axis = axis_marker()

  def set_camera_scale(self, scale):
    self.camera_scale = scale
    for mesh, points in zip(self.camera, self.camera_projections):
      mesh.points = points * scale
    
    self.axis.points = axis_points(scale * 3)


  def update(self, calib):
    self.camera_projections = [projection_corners(camera) 
      for camera in calib.cameras]

    self.set_camera_scale(self.camera_scale)
    for mesh, board in zip(self.board, calib.boards):
      mesh.points = board.mesh.points

camera_colors = struct(
  inactive = (0.5, 0.5, 0.5),
  active_camera = (1.0, 1.0, 0.0),
  active_set = (0.0, 1.0, 0.0)
)


class AxisSet():
  def __init__(self, viewer, axis_mesh, poses):

    options = struct(rgb=True, scalars='colors', show_edges=True)
    self.instances = [
      Marker(viewer, pv.PolyData(axis_mesh), pose, options=options)
        for pose in poses._sequence()]

    for instance in self.instances:
      instance.set_point_size(0)


  def update_poses(self, poses):
    for pose, marker in zip(poses._sequence(), self.instances):
      marker.set_transform(pose=pose)

  def show(self, shown):
    for marker in self.instances: 
      marker.show(shown)


class CameraSet():
  def __init__(self, viewer, camera_poses, camera_meshes):
    options = dict(show_edges=True, ambient=0.5)
    self.instances = [Marker(viewer, mesh, pose, options) 
      for mesh, pose in zip(camera_meshes, camera_poses._sequence()) ]

  def update_poses(self, camera_poses):
    for pose, marker in zip(camera_poses._sequence(), self.instances):
      marker.set_transform(pose=pose)

  def update(self, highlight, active=True):
      for i, marker in enumerate(self.instances):
        color = (camera_colors.inactive if not active
          else camera_colors.active_camera if i == highlight 
          else camera_colors.active_set)

        marker.set_color(color, 0.2 if not active else 1.0)

  def show(self, shown):
    for marker in self.instances: 
      marker.show(shown)

class BoardSet():
  def __init__(self, viewer, board_poses, board_meshes, board_colors):

    def instance(mesh, pose, color):
      options = dict(style="wireframe", color=color, show_edges=True)
      return Marker(viewer, mesh, pose, options)

    self.board_colors = board_colors
    self.instances = [instance(mesh, pose, color)
        for mesh, color, pose in zip(board_meshes, board_colors, board_poses._sequence())]

    self.update(True)

  def update_poses(self, board_poses):
    for instance, pose in zip(self.instances, board_poses._sequence()):
      instance.set_transform(pose=pose)
      

  def update(self, active):
    inactive = (0.5, 0.5, 0.5)
    for board_color, marker in zip(self.board_colors, self.instances):
        color = board_color if active else inactive
        opacity = 1 if active else 0.1
        ambient = 0.8 if active else 0.0

        marker.set_color(color, opacity)
        marker.set_ambient(ambient)

  def show(self, shown):
    for marker in self.instances: 
      marker.show(shown)