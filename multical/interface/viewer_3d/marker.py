from multical import tables
from structs.numpy import shape
from structs.struct import choose, struct
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

def projection_corners(camera):
  return np.concatenate([
      np.zeros((1, 3)),
      view_projection(camera, size=1)
  ])

def axis_marker():
  red =   [1, 0, 0]
  green = [0, 1, 0]
  blue =  [0, 0, 1]

  colors = np.array([red, red, green, green, blue, blue])

  origin = [0, 0, 0]
  corners = np.array([
      origin, [1, 0, 0], origin, [0, 1, 0], origin, [0, 0, 1],
    ]
  )

  lines = np.array([
    [2, 0, 1],
    [2, 2, 3],
    [2, 3, 4]
  ])

  mesh = pv.PolyData(corners, lines)
  mesh['colors'] = colors
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
    self.actor = viewer.add_mesh(mesh, **options)
    self.showing = True
    self.set_transform(pose, scale)

  def set_transform(self, pose, scale):
    self.pose = pose
    self.scale = scale

    transform = scaled(self.pose.poses, self.scale)
    self.actor.SetUserTransform(vtk_transform(transform))
    self.actor.SetVisibility(self.pose.valid and self.showing)

  def set_color(self, color, opacity=1):
    p = self.actor.GetProperty()
    p.SetColor(*color)
    p.SetOpacity(opacity)    


  def show(self, shown):
    self.showing = shown
    self.actor.SetVisibility(self.pose.valid and self.showing)

   
class SceneMeshes():
  def __init__(self, calib):
    self.board =  [pv.PolyData(board.mesh.points, board.mesh.polygons)
       for board in calib.boards]

    self.camera = [pv.PolyData(projection_corners(camera), view_triangles) 
      for camera in calib.cameras]


  def update(self, calib):
    for mesh, camera in zip(self.camera, calib.cameras):
      mesh.points = projection_corners(camera)

    for mesh, board in zip(self.board, calib.boards):
      mesh.points = board.mesh.points

camera_colors = struct(
  inactive = (0.5, 0.5, 0.5),
  active_camera = (1.0, 1.0, 0.0),
  active_set = (0.0, 1.0, 0.0)
)




class CameraSet():
  def __init__(self, viewer, camera_poses, camera_meshes, scale):
    poses = tables.inverse(camera_poses)
    options = dict(show_edges=True)
    self.instances = [Marker(viewer, mesh, pose, options, scale) 
      for mesh, pose in zip(camera_meshes, poses._sequence()) ]

    # self.axis_mesh = axis_marker()
    # self.axis_markers = [Marker(viewer, pv.PolyData(self.axis_mesh), )


  def update_poses(self, camera_poses):
    poses = tables.inverse(camera_poses)
    for pose, marker in zip(poses._sequence(), self.instances):
      marker.update(pose=pose)

  def update(self, highlight, scale, active=True):
      for i, marker in enumerate(self.instances):
        color = (camera_colors.inactive if not active
          else camera_colors.active_camera if i == highlight 
          else camera_colors.active_set)
        marker.set_transform(pose=marker.pose, scale=scale)
        marker.set_color(color)

  def show(self, shown):
    for marker in self.instances: 
      marker.show(shown)

class BoardSet():
  def __init__(self, viewer, board_poses, board_meshes, board_colors):

    def instance(mesh, pose, color):
      options = dict(style="wireframe", ambient=0.5, color=color, show_edges=True)
      return Marker(viewer, pv.PolyData(mesh), pose, options)

    self.board_colors = board_colors
    self.instances = [instance(mesh, pose, color)
        for mesh, color, pose in zip(board_meshes, board_colors, board_poses._sequence())]

  def update_poses(self, board_poses):
    for board_frames, board_poses in zip(self.instances,  board_poses._sequence()):
      for marker, pose in zip(board_frames, board_poses._sequence()):
        marker.update(pose=pose)
      

  def update(self, active):
    inactive = (0.5, 0.5, 0.5)
    for board_color, marker in zip(self.board_colors, self.instances):
        color = board_color if active else inactive
        opacity = 1 if active else 0.1
        marker.set_color(color, opacity)

  def show(self, shown):
    for marker in self.instances: 
      marker.show(shown)