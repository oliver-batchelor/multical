from multical.interface.moving_board import MovingBoard
from multical.interface.camera_view import CameraView
import PyQt5.QtWidgets as QtWidgets

from PyQt5 import uic
from PyQt5.QtCore import Qt


import numpy as np
from .vtk_tools import *

from multical import image

from .viewer import Viewer
from .moving_cameras import MovingCameras

def visualize(calib, images, camera_names, image_names):
  app = QtWidgets.QApplication([])

  vis = Visualizer()

  vis.update(calib, images, camera_names, image_names)
  vis.showMaximized()

  app.exec_()
  

class Visualizer(QtWidgets.QMainWindow):
  def __init__(self):
    super(Visualizer, self).__init__()
    uic.loadUi('multical/interface/visualizer.ui', self)

    self.grabKeyboard()

    self.viewer = Viewer(self)
    self.view_frame.setLayout(h_layout(self.viewer))
    self.scene = None
    self.sizes = struct(camera=0, frame=0)
    self.controller = None

    self.connect_ui()
    self.setDisabled(True)
    
  def update(self, calib, images, camera_names, image_names):
    self.image_names = image_names
    self.calib = calib
    self.images = images
    self.undistorted = image.undistort.undistort_images(images, calib.cameras)

    self.viewer.clear()

    self.camera_combo.clear()
    self.camera_combo.addItems(camera_names)
    self.frame_slider.setMaximum(len(image_names) - 1)

    self.sizes = struct(frame=len(image_names), camera=len(camera_names))
   
    self.controllers = struct(
      moving_cameras = MovingCameras(self.viewer, calib),
      moving_board = MovingBoard(self.viewer, calib),
      camera_view = CameraView(self.viewer, calib, self.undistorted)
    )

    self.viewer.fix_camera()
    self.update_controller()
    
    self.setDisabled(False)


  @property
  def camera_size(self):
    return self.camera_size_slider.value() / 500.0



  def keyPressEvent(self, event):
    if event.key() == Qt.Key.Key_Left:
      self.move_frame(-1)
    elif event.key() == Qt.Key.Key_Right:
      self.move_frame(1)
    elif event.key() == Qt.Key.Key_Up:
      self.move_camera(-1)
    elif event.key() == Qt.Key.Key_Down:
      self.move_camera(1)
    elif event.key() == Qt.Key.Key_Plus:
      self.point_size_slider.setValue(self.point_size_slider.value() + 1)
      self.line_size_slider.setValue(self.line_size_slider.value() + 1)

    elif event.key() == Qt.Key.Key_Minus:
      self.point_size_slider.setValue(self.point_size_slider.value() - 1)
      self.line_size_slider.setValue(self.line_size_slider.value() - 1)




  @property
  def state(self):
    return struct(
      frame=self.frame_slider.sliderPosition(), 
      camera=self.camera_combo.currentIndex(),
      scale=self.camera_size)

  def move_frame(self, d_frame=0):
    frame_index = (self.frame_slider.sliderPosition() + d_frame) % self.sizes.frame
    self.frame_slider.setValue(frame_index)

  def move_camera(self, d_camera=0):
    camera_index = (self.camera_combo.currentIndex() + d_camera) % self.sizes.camera
    self.camera_combo.setCurrentIndex(camera_index)

  def update_frame(self):
    if self.controller is not None:
      self.controller.update(self.state)

  def update_controller(self):
    if self.controller is not None:
      self.controller.disable()

    if self.moving_cameras.isChecked():
      self.controller = self.controllers.moving_cameras
    elif self.moving_board.isChecked():
      self.controller = self.controllers.moving_board
    else:
      self.controller = self.controllers.camera_view

    self.controller.enable(self.state)

  def connect_ui(self):
    self.camera_size_slider.valueChanged.connect(self.update_frame)

    self.frame_slider.valueChanged.connect(self.update_frame)
    self.camera_combo.currentIndexChanged.connect(self.update_frame)


    self.point_size_slider.valueChanged.connect(self.viewer.set_point_size)
    self.line_size_slider.valueChanged.connect(self.viewer.set_line_size)

    self.view_mode.buttonClicked.connect(self.update_controller)

def h_layout(*widgets):
  layout = QtWidgets.QHBoxLayout()
  for widget in widgets:
    layout.addWidget(widget)

  return layout