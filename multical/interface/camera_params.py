import PyQt5.QtWidgets as QtWidgets
from PyQt5 import QtCore

from PyQt5 import uic
from PyQt5.QtCore import Qt
import numpy as np


def set_master(poses, index):
  inv = np.linalg.inv(poses.poses[index])
  return poses._extend(poses = poses.poses @ np.expand_dims(inv, 0))

class ParamsViewer(QtWidgets.QScrollArea):
  def __init__(self):
    super().__init__(self)


  def init(self, camera_names):
    self.master_combo = QtWidgets.QComboBox(self)
    self.master_combo.currentIndexChanged.connect(self.update_master)
    self.master_combo.addItems(camera_names)

    camera_widgets = [CameraParams(name) for name in camera_names]
    layout = QtWidgets.QVBoxLayout()

    header = QtWidgets.QHBoxLayout()
    header.addItem(self.master_combo)
    header.addStretch()

    layout.addItem(header)

    for widget in camera_widgets:
      layout.addItem(widget)

    self.camera_poses = None
    self.cameras = None

    layout.addStretch()
    self.setLayout(layout)

    self.setDisabled(True)
    self.master_combo.currentIndexChanged(self.update_cameras)

  def set_cameras(self, cameras, camera_poses):
    self.cameras = cameras
    self.camera_poses = camera_poses

  def update_cameras(self):
    master = self.master_combo.getCurrentIndex()
    poses = set_master(self.camera_poses, master)

    for camera_widget, camera, camera_pose in\
       zip(self.camera_widgets, self.cameras, poses.pose):
      camera_widget.set_camera(camera)
      camera_widget.set_pose(camera_pose)

    self.setDisabled(False)

    

class CameraParams(QtWidgets.QWidget):
  def __init__(self, camera_name, parent=None):
    super(CameraParams, self).__init__(parent)
    uic.loadUi('multical/interface/camera.ui', self)

    self.groupBox.setTitle(f"{camera_name}")
    self.camera_name = camera_name

  def set_camera(self, camera):
    w, h = camera.image_size
    self.groupBox.setTitle(f"{self.camera_name} ({w}x{h})")

    for i in range(3):
      for j in range(3):
        v = camera.intrinsic[i, j]
        self.intrinsic_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:.4f}"))

    self.dist_table.setColumnCount(camera.dist.size)
    for i in range(camera.dist.size):
      v = camera.dist[0, i]
      self.dist_table.setItem(0, i, QtWidgets.QTableWidgetItem(f"{v:.4f}"))


  def set_pose(self, camera_pose):

    for i in range(3):
      for j in range(4):
        v = camera_pose[i, j]
        self.extrinsic_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:.4f}"))