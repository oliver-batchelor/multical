
from collections import OrderedDict
from functools import partial
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFont
from qtpy.QtCore import QLineF, QPointF, QRectF, Qt
from cached_property import cached_property

import cv2
import numpy as np

from structs.struct import choose, struct, transpose_lists
from structs.numpy import Table, shape

def qt_image(image):
  assert image.ndim in [2, 3], f"qt_image: unsupported image dimensions {image.ndim}"
  assert image.dtype == np.uint8, f"qt_image: expected np.uint8 got {image.dtype}"

  if image.ndim == 3:
    height, width, _ = image.shape
    bytesPerLine = 3 * width
    return QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
  elif image.ndim == 2:
    height, width = image.shape
    bytesPerLine = width
    return QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
  

def cosmetic_pen(color, width=2):
    pen = QPen(QColor.fromRgbF(*color))
    pen.setCosmetic(True)
    pen.setWidth(width)
    return pen


class Lazy(object):
  def __init__(self, f, *args, **kwargs):
    self.compute = partial(f, *args, **kwargs)

  @cached_property
  def value(self):
    return self.compute()


def group(items):
  itemGroup = QtWidgets.QGraphicsItemGroup()
  for item in items:
    itemGroup.addToGroup(item)
  return itemGroup

def line(p1, p2, pen):
  line = QtWidgets.QGraphicsLineItem(*p1, *p2)
  line.setPen(pen)
  return line


def cross(point, radius, pen):
  x, y = point
  item = group([
    line([-radius, 0], [radius, 0], pen), 
    line([0, -radius], [0, radius], pen)
  ])
  item.setPos(x, y)
  return item

def id_marker(id, point, color, font):
  item = QtWidgets.QGraphicsTextItem(str(id))
  item.setDefaultTextColor(QColor.fromRgbF(*color))
  item.setFont(font)
  
  container = group([item])

  x, y = point
  container.setPos(QPointF(x - 2, y - 2))
  return container


colors = struct(
  error_line = (1, 1, 0),
  outlier =    (1, 0, 0),
  inlier =     (0, 1, 0),
  invalid =    (0.5, 0.5, 0))


def add_marker(scene, corner, id, options, pen, color, font):
    scene.addItem(cross(corner, options.marker_size, pen))
    if options.show_ids:
      scene.addItem(id_marker(id, corner,  color, font))



def add_point_markers(scene, points, board, color, options):
  marker_font = QFont()
  marker_font.setPixelSize(options.marker_size * 0.75)

  pen = cosmetic_pen(color, options.line_width)
  corners = points.points[points.valid]

  for corner, id in zip(corners, board.ids[points.valid]):
    add_marker(scene, corner, id, options, pen, color, marker_font)

def add_reprojections(scene, points, projected, inliers, boards, valid_boards, options):
  marker_font = QFont()
  marker_font.setPixelSize(int(options.marker_size * 0.75))

  frame_table = points._extend(proj=projected.points, inlier=inliers)

  colors = struct(
    error_line = (1, 1, 0),
    outlier =    (1, 0, 0),
    inlier =     (0, 1, 0),
    invalid =    (0.5, 0.5, 0))  
  
  pens = colors._map(cosmetic_pen, options.line_width)

  for board, valid_board, board_points in zip(boards, valid_boards, frame_table._sequence(0)):
      if not valid_board:
        continue

      for point, id in zip(board_points._sequence(), board.ids):

        color_key = 'invalid' if not point.valid else ('inlier' if point.inlier else 'outlier')
        add_marker(scene, point.proj, id, options, pens[color_key], colors[color_key], marker_font)

        if point.valid:
          scene.addItem(line(point.proj, point.points, pens.error_line))


def annotate_image(workspace, calibration, layer, state, options):

  image = state.image

  scene = QtWidgets.QGraphicsScene()
  pixmap = QPixmap(qt_image(image))
  scene.addPixmap(pixmap)

  detections = workspace.point_table._index[state.camera, state.frame]
  if layer == "detections":

    for board, color, points in zip(workspace.boards, workspace.board_colors, detections._sequence(0)):
      add_point_markers(scene, points, board, color, options)

  elif layer == "reprojection":

    assert calibration is not None
    projected = calibration.projected._index[state.camera, state.frame]
    inliers = calibration.inliers[state.camera, state.frame]
    valid_boards = calibration.pose_estimates.board.valid

    add_reprojections(scene, detections, projected, inliers, workspace.boards, valid_boards, options)

  elif layer == "detected_poses":
    board_poses = workspace.pose_table._index[state.camera, state.frame]
    camera = workspace.initialisation.cameras[state.camera]

    for board, pose, color in zip(workspace.boards, board_poses._sequence(0), workspace.board_colors):
      if pose.valid:
        projected = Table.create(
          points = camera.project(board.points, pose.poses),
          valid = np.ones(board.points.shape[0], dtype=np.bool)
        )
        add_point_markers(scene, projected, board, color, options)

  else:
    assert False, f"unknown layer {layer}"

  h, w, *_ = image.shape
  scene.setSceneRect(-w, -h, 3 * w, 3 * h)  
  return scene  

class ViewerImage(QtWidgets.QGraphicsView):
  def __init__(self, parent):
    super(ViewerImage, self).__init__(parent)

    self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
    self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
    self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
    self.setFrameShape(QtWidgets.QFrame.NoFrame)
    self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    self.zoom = 0
    self.zoom_factor = 0.95
    self.zoom_range = (-30, 40)

  def update_image(self, scene): 
    self.setScene(scene)
    self.update()


  def wheelEvent(self, event):
    zooming_in = event.angleDelta().y() < 0
    new_zoom = self.zoom - (zooming_in * 2 - 1)
 
    if new_zoom > self.zoom_range[0] and new_zoom < self.zoom_range[1]:
      factor = self.zoom_factor if zooming_in else 1/self.zoom_factor
      self.scale(factor, factor)
      self.zoom = new_zoom

    self.update()
