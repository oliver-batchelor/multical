import PyQt5.QtWidgets as QtWidgets
from PyQt5 import QtCore

from PyQt5 import uic
from PyQt5.QtCore import Qt


def params_viewer(parent, calib, camera_names):

  layout = QtWidgets.QVBoxLayout()
  for camera_name, camera, camera_pose in\
     zip(camera_names, calib.cameras, calib.pose_estimates.camera._sequence()):

    camera_widget = CameraParams(parent)
    camera_widget.init(camera_name, camera, camera_pose)

    layout.addWidget(camera_widget)

  layout.addStretch()
  return layout

class CameraParams(QtWidgets.QWidget):
  def __init__(self, parent=None):
    super(CameraParams, self).__init__(parent)
    uic.loadUi('multical/interface/camera.ui', self)


  def init(self, camera_name, camera, camera_pose):
    w, h = camera.image_size
    self.groupBox.setTitle(f"{camera_name} ({w}x{h})")

    for i in range(3):
      for j in range(3):
        v = camera.intrinsic[i, j]
        self.intrinsic_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:.4f}"))


    for i in range(3):
      for j in range(4):
        v = camera_pose.poses[i, j]
        self.extrinsic_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:.4f}"))


    self.dist_table.setColumnCount(camera.dist.size)
    for i in range(camera.dist.size):
      v = camera.dist[0, i]
      self.dist_table.setItem(0, i, QtWidgets.QTableWidgetItem(f"{v:.4f}"))

