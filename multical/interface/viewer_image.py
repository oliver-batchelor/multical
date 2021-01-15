
from functools import partial
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFont
from PyQt5.QtCore import QLineF, QPointF, QRectF, Qt
from cached_property import cached_property

import cv2
import numpy as np

from structs.struct import choose, struct, transpose_lists

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
  

def cosmetic_pen(color):
    pen = QPen(QColor(*color))
    pen.setCosmetic(True)
    pen.setWidth(2)
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
  return group([
    line([x - radius, y], [x + radius, y], pen), 
    line([x, y - radius], [x, y + radius], pen)
  ])

def id_marker(id, point, radius, color, font):
  item = QtWidgets.QGraphicsTextItem(str(id))

  x, y = point
  item.setPos(QPointF(x + radius / 2, y + radius / 2))
  item.setDefaultTextColor(QColor(*color))
  item.setFont(font)



annotation_colors = struct(
  error_line = (255, 255, 0),
  detected = (0, 0, 255),
  outlier = (255, 0, 0),
  inlier = (255, 0, 0),
  invalid = (128, 128, 128),
  pose = (0, 192, 255))

def annotate_image(board, image, camera, image_table, radius=10.0):
  scene = QtWidgets.QGraphicsScene()

  pixmap = QPixmap(qt_image(image))
  scene.addPixmap(pixmap)

  detected = []
  refined = []
  pose = []
  ids = []
  
  pens = annotation_colors._map(cosmetic_pen)
  marker_font = QFont()
  marker_font.setPixelSize(radius / 2)

  projected_points = camera.project(board.adjusted_points, image_table.poses).astype(np.float32)
  projected_pose = camera.project(board.points, image_table.pose_detections).astype(np.float32)

  for proj, corner, valid, pose_point, inlier in\
    zip(projected_points, image_table.points, image_table.valid_points, projected_pose,  image_table.inliers):

      if valid:
        refined.append(line(proj, corner, pens.error_line))
        detected.append( cross(corner, radius, pens.detected))
        # ids.append(id_marker(corner, radius, annotation_colors.detected))

      proj_pen = pens.invalid if not valid else pens.inlier if inlier else pens.outlier
      refined.append(cross(proj, radius, proj_pen))  
      pose.append(cross(pose_point, radius, pens.pose))        


  layers = struct(detected = group(detected), refined = group(refined), 
    pose=group(pose), ids=group(ids))
  
  for layer in layers.values():
    scene.addItem(layer)

  h, w, *_ = image.shape
  scene.setSceneRect(-w / 2, -h / 2, 2 * w, 2 * h)    

  return struct(scene=scene, layers=layers, pens=pens)


def annotate_images(calib, images, radius=10.0):
  table = calib.point_table._merge(calib.pose_table)._extend(inliers = calib.inliers)

  table = table._extend(
     pose_detections = calib.pose_detections.poses, 
     valid_detection = calib.pose_detections.valid_poses)

  return [[Lazy(annotate_image, calib.board, image, camera, image_table, radius=radius) 
      for image, image_table in zip(cam_images, image_table._sequence())]
        for camera, cam_images, image_table in zip(calib.cameras, images, table._sequence())]


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

  def update_image(self, lazy_scene, layers):  
    scene = lazy_scene.value
    for name, layer in scene.layers.items():
      layer.setVisible(layers[name])

    self.setScene(scene.scene)
    self.update()


  def wheelEvent(self, event):
    zooming_in = event.angleDelta().y() < 0
    new_zoom = self.zoom - (zooming_in * 2 - 1)
 
    if new_zoom > self.zoom_range[0] and new_zoom < self.zoom_range[1]:
      factor = self.zoom_factor if zooming_in else 1/self.zoom_factor
      self.scale(factor, factor)
      self.zoom = new_zoom

    self.update()
