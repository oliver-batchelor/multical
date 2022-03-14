import qtpy.QtWidgets as QtWidgets
from qtpy import QtCore

from .ui_files import load_ui

from qtpy.QtCore import Qt
import numpy as np

from .layout import h_layout, h_stretch, v_stretch, v_layout, widget


def set_master(poses, index):
  inv = np.linalg.inv(poses.poses[index])
  return poses._extend(poses = poses.poses @ np.expand_dims(inv, 0))

class ParamsViewer(QtWidgets.QScrollArea):
  def __init__(self, parent):
    super().__init__(parent)

    self.camera_poses = None
    self.cameras = None
    self.camera_names = None



  def init(self, camera_names):
    self.camera_names = camera_names
    self.master_combo = QtWidgets.QComboBox(self)
    self.master_combo.addItems(camera_names)
   
    self.camera_widgets = [CameraParams(name) for name in camera_names]

    layout = v_layout(
      h_layout(QtWidgets.QLabel("Master", self), self.master_combo, h_stretch()),
      *self.camera_widgets,
      v_stretch())

    self.setWidget(widget(layout))
    self.setWidgetResizable(True)

    self.setDisabled(True)
    self.master_combo.currentIndexChanged.connect(self.update_cameras)

  def set_cameras(self, cameras, camera_poses):
    assert self.camera_names is not None

    self.cameras = cameras
    self.camera_poses = camera_poses

    self.update_cameras()

  def update_cameras(self):
    master = self.master_combo.currentIndex()
    poses = set_master(self.camera_poses, master)

    for camera_widget, camera, camera_pose in\
       zip(self.camera_widgets, self.cameras, poses.poses):
      camera_widget.set_camera(camera)
      camera_widget.set_pose(camera_pose)

    self.setDisabled(False)

    

class CameraParams(QtWidgets.QWidget):
  def __init__(self, camera_name, parent=None):
    super(CameraParams, self).__init__(parent)

    load_ui(self, "camera.ui")


    self.groupBox.setTitle(f"{camera_name}")
    self.camera_name = camera_name

  def set_camera(self, camera):
    w, h = camera.image_size
    self.groupBox.setTitle(f"{self.camera_name} ({w}x{h})")
    # fisheye.calibrate dist matrix is of shape (4,1) instead of (1,5)
    dist_format_rows = camera.dist.shape[0] > 1

    for i in range(3):
      for j in range(3):
        v = camera.intrinsic[i, j]
        self.intrinsic_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:.4f}"))

    self.dist_table.setColumnCount(camera.dist.size)
    for i in range(camera.dist.size):
      if dist_format_rows:
        v = camera.dist[i, 0]
      else:
        v = camera.dist[0, i]
      self.dist_table.setItem(0, i, QtWidgets.QTableWidgetItem(f"{v:.4f}"))


  def set_pose(self, camera_pose):

    for i in range(3):
      for j in range(4):
        v = camera_pose[i, j]
        self.extrinsic_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:.4f}"))