import math
from multical.interface import view_table
from multical.interface.moving_board import MovingBoard
from multical.interface.camera_view import CameraView
from multical.interface import camera_params
import PyQt5.QtWidgets as QtWidgets

from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt

from multical import image, tables

from .vtk_tools import *
from .viewer_3d import Viewer3D
from .viewer_image import ViewerImage, annotate_images
from .moving_cameras import MovingCameras



def visualize(calib, images, camera_names, image_names):
  app = QtWidgets.QApplication([])

  vis = Visualizer()

  print("Undistorting images...")
  undistorted = image.undistort.undistort_images(images, calib.cameras)

  vis.init(calib, images, undistorted, camera_names, image_names)
  vis.showMaximized()

  app.exec_()


class Visualizer(QtWidgets.QMainWindow):
  def __init__(self):
    super(Visualizer, self).__init__()
    uic.loadUi('multical/interface/visualizer.ui', self)

    self.setFocusPolicy(Qt.StrongFocus)
    self.grabKeyboard()

    self.viewer_3d = Viewer3D(self)
    self.viewer_image = ViewerImage(self)

    self.view_frame.setLayout(h_layout(self.viewer_3d))
    self.image_frame.setLayout(h_layout(self.viewer_image))

    self.scene = None
    self.sizes = struct(camera=0, frame=0)
    self.controller = None
    self.splitter.setStretchFactor(0, 10)
    self.splitter.setStretchFactor(1, 1)

    self.setDisabled(True)

  def init(self, calib, images, undistorted, camera_names, image_names):

    self.calib = calib
    self.images = images
    self.camera_names = camera_names
    self.image_names = image_names
    self.undistorted = undistorted

    self.blockSignals(True)
    self.viewer_3d.clear()

    self.annotated_images = annotate_images(calib, images)

    self.view_model = view_table.ViewModel(calib, camera_names, image_names)
    self.metric_combo.clear()
    self.metric_combo.addItems(self.view_model.metric_labels)

    self.view_table.setSelectionMode(
        QtWidgets.QAbstractItemView.SingleSelection)
    self.view_table.setModel(self.view_model)
    self.select(0, 0)

    self.sizes = struct(frame=len(image_names), camera=len(camera_names))

    self.controllers = struct(
        moving_cameras=MovingCameras(self.viewer_3d, calib),
        moving_board=MovingBoard(self.viewer_3d, calib),
        camera_view=CameraView(self.viewer_3d, calib, self.undistorted)
    )

    params_layout = camera_params.params_viewer(self, calib, camera_names)
    self.param_scroll.setLayout(params_layout)

    self.viewer_3d.fix_camera()
    self.update_controller()
    self.update_frame()

    self.setDisabled(False)
    self.blockSignals(False)

    self.connect_ui()

  def keyPressEvent(self, event):

    if event.key() == Qt.Key.Key_Up:
      self.move_frame(-1)
    elif event.key() == Qt.Key.Key_Down:
      self.move_frame(1)
    elif event.key() == Qt.Key.Key_Left:
      self.move_camera(-1)
    elif event.key() == Qt.Key.Key_Right:
      self.move_camera(1)
    elif event.key() == Qt.Key.Key_Plus:
      self.point_size_slider.setValue(self.point_size_slider.value() + 1)
      self.line_size_slider.setValue(self.line_size_slider.value() + 1)

    elif event.key() == Qt.Key.Key_Minus:
      self.point_size_slider.setValue(self.point_size_slider.value() - 1)
      self.line_size_slider.setValue(self.line_size_slider.value() - 1)

    self.update()

  def camera_size(self):
    return self.camera_size_slider.value() / 500.0

  def select(self, frame, camera):
    index = self.view_model.index(frame, camera)
    self.view_table.selectionModel().select(
        index, QtCore.QItemSelectionModel.ClearAndSelect)

  def selection(self):
    selection = self.view_table.selectionModel().selectedIndexes()
    assert len(selection) == 1

    return selection[0].row(), selection[0].column()

  def state(self):
    frame, camera = self.selection()

    return struct(
        frame=frame,
        camera=camera,
        scale=self.camera_size(),
        camera_name=self.camera_names[camera],
        image_name=self.image_names[frame],
        image=self.images[camera][frame]

    )

  def move_frame(self, d_frame=0):
    frame, camera = self.selection()
    self.select((frame + d_frame) % self.sizes.frame, camera)
  
  def move_camera(self, d_camera=0):
    frame, camera = self.selection()
    self.select(frame, (camera + d_camera) % self.sizes.camera)

  def update_frame(self):
    state = self.state()
    if self.controller is not None:
      self.controller.update(state)

    self.statusBar().showMessage(f"{state.camera_name} {state.image_name}")
    self.update_image()

  def update_image(self):
    state = self.state()
    annotated_image = self.annotated_images[state.camera][state.frame]
    image_layers = struct(
        refined=self.show_refined_check.isChecked(),
        detected=self.show_detected_check.isChecked(),
        ids=self.show_ids_check.isChecked(),
        pose=self.show_pose_check.isChecked()
    )

    self.viewer_image.update_image(annotated_image, image_layers)

  def update_controller(self):
    if self.controller is not None:
      self.controller.disable()

    if self.moving_cameras_radio.isChecked():
      self.controller = self.controllers.moving_cameras
    elif self.moving_board_radio.isChecked():
      self.controller = self.controllers.moving_board
    else:
      self.controller = self.controllers.camera_view

    self.controller.enable(self.state())

  def update_table(self):
    inlier_only = self.inliers_check.isChecked()
    metric_selected = self.metric_combo.currentIndex()

    self.view_model.set_metric(metric_selected, inlier_only)


  def connect_ui(self):
    self.camera_size_slider.valueChanged.connect(self.update_frame)
    self.view_table.selectionModel().selectionChanged.connect(self.update_frame)

    self.inliers_check.toggled.connect(self.update_table)
    self.metric_combo.currentIndexChanged.connect(self.update_table)

    for layer_check in [self.show_refined_check, self.show_detected_check, self.show_ids_check, self.show_pose_check]:
      layer_check.toggled.connect(self.update_image)

    self.marker_size_slider.valueChanged.connect(self.update_image)

    self.point_size_slider.valueChanged.connect(self.viewer_3d.set_point_size)
    self.line_size_slider.valueChanged.connect(self.viewer_3d.set_line_size)

    self.view_mode.buttonClicked.connect(self.update_controller)


def h_layout(*widgets):
  layout = QtWidgets.QHBoxLayout()
  for widget in widgets:
    layout.addWidget(widget)

  return layout
