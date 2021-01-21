import math

from .viewer_3d.viewer_3d import Viewer3D
from .viewer_3d.moving_board import MovingBoard
from .viewer_3d.camera_view import CameraView
from .viewer_3d.moving_cameras import MovingCameras

from . import camera_params, view_table
import PyQt5.QtWidgets as QtWidgets

from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt

from multical import image, tables

from .viewer_image import ViewerImage, annotate_image
from structs.struct import struct, split_dict


def visualize(workspace):
  app = QtWidgets.QApplication([])

  vis = Visualizer()

  vis.update_workspace(workspace)
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

    self.controller = None
    self.splitter.setStretchFactor(0, 10)
    self.splitter.setStretchFactor(1, 1)

    self.setDisabled(True)

  def update_calibrations(self, calibrations):
    has_calibrations = len(calibrations) > 0

    self.tab_3d.setEnabled(has_calibrations)
    self.cameras_tab.setEnabled(has_calibrations)  

    self.viewer_3d.clear()
    self.controllers = None
    self.calibration = None
 
    if has_calibrations:
      names, calibs = split_dict(calibrations)
      self.set_calibration(calibs[-1])

    self.setup_view_table(self.calibration)

    _, layer_labels = self.image_layers()
    self.layer_combo.clear()
    self.layer_combo.addItems(layer_labels)
      
  def set_calibration(self, calibration):
    self.calibration = calibration

    params_layout = camera_params.params_viewer(self, self.calibration.cameras, self.workspace.camera_names)
    self.param_scroll.setLayout(params_layout)

    self.controllers = struct(
        moving_cameras=MovingCameras(self.viewer_3d, self.calibration),
        moving_board=MovingBoard(self.viewer_3d, self.calibration)
    )

    self.viewer_3d.fix_camera()
    self.update_controller()        


  def setup_view_table(self, calibration):
    board_labels = ["All boards"] + [f"Board: {name}" for name in self.workspace.names.board]

    self.boards_combo.clear()
    self.boards_combo.addItems(board_labels)

    self.view_model = view_table.ViewModelDetections(self.workspace.point_table, self.workspace.names)\
      if calibration is None else view_table.ViewModelCalibrated(calibration, self.workspace.names) 

    self.metric_combo.clear()
    self.metric_combo.addItems(self.view_model.metric_labels)

    self.view_table.setSelectionMode(
        QtWidgets.QAbstractItemView.SingleSelection)
    self.view_table.setModel(self.view_model)
    self.select(0, 0)



  def update_workspace(self, workspace):

    self.workspace = workspace
    self.blockSignals(True)

    self.calibrations = self.workspace.get_calibrations()
    self.update_calibrations(self.calibrations)

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
    names = self.workspace.names
    images = self.workspace.images

    return struct(
        frame=frame,
        camera=camera,
        scale=self.camera_size(),
        camera_name=names.camera[camera],
        image_name=names.image[frame],
        image=images[camera][frame]
    )

  def move_frame(self, d_frame=0):
    sizes = self.workspace.sizes 

    frame, camera = self.selection()
    self.select((frame + d_frame) % sizes.image, camera)
  
  def move_camera(self, d_camera=0):
    sizes = self.workspace.sizes 

    frame, camera = self.selection()
    self.select(frame, (camera + d_camera) % sizes.camera)

  def update_frame(self):
    state = self.state()
    if self.controller is not None:
      self.controller.update(state)

    self.statusBar().showMessage(f"{state.camera_name} {state.image_name}")
    self.update_image()

  def image_layers(self):
    layers = struct(detections="Detections")
    if self.calibration is not None:
      layers.reprojection = "Reprojection"
    
    if self.workspace.detected_poses is not None:
      layers.detected_poses = "Detected poses"
      
    return split_dict(layers)

  def update_image(self):
    state = self.state()
    layer_names, _ = self.image_layers()


    layer_name = layer_names[self.layer_combo.currentIndex()]

    options = struct(
      marker_size = self.marker_size_slider.value(),
      line_width = self.line_width_slider.value(),
      show_ids=self.show_ids_check.isChecked()
    )

    annotated_image = annotate_image(self.workspace, self.calibration, 
      layer_name, state, options=options)

    self.viewer_image.update_image(annotated_image)

  def update_controller(self):
    if self.controller is not None:
      self.controller.disable()

    if self.moving_cameras_check.isChecked():
      self.controller = self.controllers.moving_cameras
    else:
      self.controller = self.controllers.camera_view

    self.controller.enable(self.state())

  def update_view_table(self):
    inlier_only = self.inliers_check.isChecked()
    metric_selected = self.metric_combo.currentIndex()
    board_index = self.boards_combo.currentIndex()
    board = None if board_index == 0 else board_index - 1

    self.view_model.set_metric(metric_selected, board, inlier_only)


  def connect_ui(self):
    self.camera_size_slider.valueChanged.connect(self.update_frame)
    self.view_table.selectionModel().selectionChanged.connect(self.update_frame)

    self.inliers_check.toggled.connect(self.update_view_table)
    self.metric_combo.currentIndexChanged.connect(self.update_view_table)
    self.boards_combo.currentIndexChanged.connect(self.update_view_table)


    for layer_check in [self.show_ids_check]:
      layer_check.toggled.connect(self.update_image)
    self.layer_combo.currentIndexChanged.connect(self.update_image)
    self.marker_size_slider.valueChanged.connect(self.update_image)
    self.line_width_slider.valueChanged.connect(self.update_image)


    self.line_size_slider.valueChanged.connect(self.viewer_3d.set_line_size)
    self.moving_cameras_check.toggled.connect(self.update_controller)


def h_layout(*widgets):
  layout = QtWidgets.QHBoxLayout()
  for widget in widgets:
    layout.addWidget(widget)

  return layout
