from structs.numpy import shape
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

def view_marker(camera, size=1):
  corners = np.concatenate([
      np.zeros((1, 3)),
      view_projection(camera, size=size),
  ])

  mesh = pv.PolyData(corners, view_triangles)
  return mesh


def image_projection(camera, distance=1):
  corners = view_projection(camera, size=distance)
  triangles = np.array([[4, 2, 1, 0, 3]])

  uvs = np.array([(0, 1), (1, 1), (1, 0), (0, 0)])

  mesh = pv.PolyData(corners, triangles)
  mesh.t_coords = uvs
  return mesh
 

class View():
  def __init__(self, viewer, camera, pose, scale=1):
    self.pose = pose
    self.camera = camera

    self.mesh = view_marker(camera)
    self.actor = viewer.add_mesh(self.mesh, show_edges=True)

    self.set_scale(scale)

  def set_scale(self, scale):
    scaling = np.diag([scale, scale, scale, 1.0])
    transform = self.pose @ scaling 

    self.actor.SetUserTransform(vtk_transform(transform))

  def set_color(self, color):
    self.actor.GetProperty().SetColor(*color)

  def show(self, shown):
    self.actor.SetVisibility(shown)



def board_mesh(board):
  corners = board.adjusted_points
  w, h = board.size

  indices = np.arange(corners.shape[0]).reshape(h - 1, w - 1)
  quad = np.array([indices[0, 0], indices[1, 0], indices[1, 1], indices[0, 1]])
  offsets = indices[: h - 2, :w - 2]

  quads = quad.reshape(1, 4) + offsets.reshape(-1, 1)
  quads = np.concatenate([np.full( (quads.shape[0], 1), 4), quads], axis=1)
  return pv.PolyData(corners, quads)


def board_object(viewer, board, color, transform=None):
  mesh = board_mesh(board)
  return viewer.add_mesh(mesh, style="wireframe", lighting=False, 
    transform=transform, color=color, show_edges=True)