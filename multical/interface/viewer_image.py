
from functools import partial
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFont
from PyQt5.QtCore import QLineF, QPointF, QRectF, Qt
from cached_property import cached_property

import cv2
import numpy as np

import palettable.colorbrewer.qualitative as palettes
from structs.struct import choose, struct, transpose_lists
from structs.numpy import shape

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
    pen = QPen(QColor(*color))
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

def id_marker(id, point, radius, color, font):
  item = QtWidgets.QGraphicsTextItem(str(id))
  item.setDefaultTextColor(QColor(*color))
  item.setFont(font)
  
  container = group([item])
  container.setPos(QPointF(*point))
  return container


colors = struct(
  error_line = (255, 255, 0),
  outlier = (255, 0, 0),
  inlier = (0, 255, 0),
  invalid = (128, 128, 0))

def board_colors(boards):
  n_colors = min(len(boards), 4)
  return getattr(palettes, f"Set1_{n_colors}").colors
  


def add_detections(scene, boards, image_table, options):
  marker_font = QFont()
  marker_font.setPixelSize(options.marker_size)
  palette = board_colors(boards)

  for board, color, detected in zip(boards, palette, image_table._sequence(0)):
    pen = cosmetic_pen(color, options.line_width)
    corners = detected.points[detected.valid_points]
    ids = board.ids[detected.valid_points]

    for corner, id in zip(corners, ids):
      scene.addItem(cross(corner, options.marker_size, pen))

    if options.show_ids:
      for corner, id in zip(corners, ids):
        scene.addItem(id_marker(id, corner, options.marker_size, color, marker_font))



# def calibrated_layers(camera, boards, image_table, radius=10):
#   refined, pose = [], []
#   pens = colors._map(cosmetic_pen)

#   projected_points = camera.project(board.adjusted_points, image_table.poses).astype(np.float32)
#   projected_pose = camera.project(board.points, image_table.pose_detections).astype(np.float32)

#   iter =  zip(projected_points, image_table.valid_points, projected_pose, image_table.inliers)

#   for proj, corner, valid, pose_point, inlier, id in iter:
#       if valid:
#         refined.append(line(proj, corner, pens.error_line))

#       proj_pen = pens.invalid if not valid else (pens.inlier if inlier else pens.outlier)
#       refined.append(cross(proj, radius, proj_pen))  
#       pose.append(cross(pose_point, radius, pens.pose))        

#   return struct(refined = group(refined), pose=group(pose))


def annotate_image(workspace, calibration, layer, state, options):

  image = state.image

  scene = QtWidgets.QGraphicsScene()
  pixmap = QPixmap(qt_image(image))
  scene.addPixmap(pixmap)

  if layer == "detections":
    detections = workspace.point_table._index[state.camera, state.frame]
    add_detections(scene, workspace.boards, detections, options)
  else:
    assert False, f"unknown layer {layer}"

  # if layer == "reprojection":
  #   assert calibration is not None
  #   camera = workspace.cameras[state.camera]

  #   detected = workspace.pose_detections._index[state.camera, state.frame]
  #   initial = workspace.pose_estimates[state.camera, state.frame]
  #   calibrated = calibration.pose_table[state.camera, state.frame]

  #   annotate_image()


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
