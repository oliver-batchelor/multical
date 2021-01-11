
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QRectF, Qt

import cv2
import numpy as np

from structs.struct import choose, transpose_lists

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
  



# def annotate_projection(board, image, camera, image_table, radius=20.0, thickness=1, scale=1):
  
#   projected = camera.project(board.adjusted_points, image_table.poses).astype(np.float32)

#   image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#   image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))

#   for proj, corner, valid, inlier in zip(projected * scale, 
#       image_table.points * scale, image_table.valid_points, image_table.inliers):

#     color = (255, 255, 0) if not valid\
#       else (0, 255, 0) if inlier else (0, 0, 255) 

#     cv2.drawMarker(image, tuple(proj), color=color, thickness=int(1), markerSize=int(radius * scale), line_type=cv2.LINE_AA)
#     if valid:
#       cv2.line(image, tuple(proj), tuple(corner), color=(0, 128, 192), thickness=thickness, lineType=cv2.LINE_AA)   


#   rvec, tvec = rtvec.split(rtvec.from_matrix(image_table.poses))
#   cv2.aruco.drawAxis(image, camera.scale_image(scale).intrinsic, 
#     camera.dist, rvec, tvec, board.square_length * 3 * scale)

#   return image



def annotate_image(board, image, camera, table, scale=20.0):
  scene = QtWidgets.QGraphicsScene()

  pixmap = QPixmap(qt_image(image))
  image = QtWidgets.QGraphicsPixmapItem(pixmap)
  scene.addItem(image)

  
  return scene



def annotate_images(calib, images, scale=20.0):
  table = calib.point_table._merge(calib.pose_table)._extend(inliers = calib.inliers)

  return [[annotate_image(calib.board, image, camera, image_table, scale=scale) 
      for image, image_table in zip(cam_images, image_table._sequence())]
        for camera, cam_images, image_table in zip(calib.cameras, images, table._sequence())]



class ViewerImage(QtWidgets.QGraphicsView):

  def __init__(self, parent):
    super(ViewerImage, self).__init__(parent)

    self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
    self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
    self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
    self.setFrameShape(QtWidgets.QFrame.NoFrame)


    self.zoom = 0
    self.zoom_factor = 0.95
    self.zoom_range = (-25, 25)

  def setImage(self, scene):  
    self.setScene(scene)
    self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    self.update()


  def wheelEvent(self, event):
    zooming_in = event.angleDelta().y() < 0
    new_zoom = self.zoom + (zooming_in * 2 - 1)
 
    if new_zoom > self.zoom_range[0] and new_zoom < self.zoom_range[1]:
      factor = self.zoom_factor if zooming_in else 1/self.zoom_factor
      self.scale(factor, factor)
      self.zoom = new_zoom

    self.update()
