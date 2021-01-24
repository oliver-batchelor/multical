import math
from os import path

from PyQt5.QtGui import QBrush, QColor
from multical import tables

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QAbstractTableModel
import numpy as np

from structs.struct import struct
from structs.numpy import shape, Table

from colour import Color


def masked_quantile(error, mask, quantiles, axis=None):
  error = error.copy()
  error[~mask] = math.nan

  return np.nanquantile(error, quantiles, axis=axis)


def reprojection_statistics(error, valid, inlier, axis=None):
  n = valid.sum(axis=axis)
  mse = np.square(error).sum(axis=axis) / np.maximum(n, 1)

  outliers = (valid & ~inlier).sum(axis=axis)
  quantiles = masked_quantile(
      error, valid, [0, 0.25, 0.5, 0.75, 1.0], axis=axis)
  min, lq, median, uq, max = quantiles

  return Table.create(detected=n, outliers=outliers, mse=mse, rms=np.sqrt(mse),
                      min=min, lower_q=lq, median=median, upper_q=uq, max=max)


sum_axes = struct(overall=None, views=(2, 3), 
  boards=(0, 1, 3), cameras=(1, 2, 3), frames=(0, 2, 3))

def reprojection_tables(calib, inlier_only=False):
  point_table = calib.point_table
  if inlier_only:
    point_table = point_table._extend(valid_points=calib.inliers)

  error, valid = tables.reprojection_error(calib.projected, point_table)

  def f(axis): return reprojection_statistics(
      error, valid, calib.inliers, axis=axis)
  return sum_axes._map(f)


def detection_tables(point_table):
  valid = point_table.valid_points
  def f(axis): return np.sum(valid, axis=axis)
  return sum_axes._map(f)


def interpolate_hsl(t, color1, color2):
  hsl1 = np.array(color1.get_hsl())
  hsl2 = np.array(color2.get_hsl())
  return Color(hsl=hsl2 * t + hsl1 * (1 - t))


class ViewModelCalibrated(QAbstractTableModel):
  def __init__(self, calib, names):
    super(ViewModelCalibrated, self).__init__()

    self.calib = calib
    self.reprojection_table = reprojection_tables(calib)
    self.inlier_table = reprojection_tables(calib, inlier_only=True)

    self.names = names

    self.metrics = dict(
        detected='Detected',
        median='Median',
        upper_q='Upper quartile',
        max='Maximum',
        rms='Root Mean Square',
        mse='Mean Square Error'
    )

    self.metric = 'detected'

  @property
  def metric_labels(self):
    return list(self.metrics.values())

  @property
  def view_table(self):
    return self.reprojection_table.views

  def cell_color(self, view_stats):
    # detection_rate = min(detection_count / self.quantiles[4], 1)

    detection_rate = min(view_stats.detected / self.calib.board.num_points, 1)
    outlier_rate = view_stats.outliers / max(view_stats.detected, 1)

    outlier_t = min(outlier_rate * 5, 1.0)
    color = interpolate_hsl(
        outlier_t, Color("green"), Color("red"))

    color.set_luminance(max(1 - detection_rate, 0.7))
    return color

  def set_metric(self, metric):
    if isinstance(metric, str):
      assert metric in self.metrics
      self.metric = metric
    else:
      assert isinstance(metric, int)
      metric_list = list(self.metrics.keys())
      self.metric = metric_list[metric]

    self.modelReset.emit()

  def data(self, index, role):
    view_stats = self.reprojection_table.views._index[index.column(), index.row()]
    inlier_stats = self.inlier_table.views._index[index.column(), index.row()]

    if role == Qt.DisplayRole:
      inlier, all = inlier_stats[self.metric], view_stats[self.metric]

      return f"{inlier} ({all})" if isinstance(inlier, int)\
        else f"{inlier:.2f} ({all:.2f})" 

    if role == Qt.BackgroundRole:
      rgb = np.array(self.cell_color(view_stats).get_rgb()) * 255
      return QBrush(QColor(*rgb))

  def headerData(self, index, orientation, role):
    if role == Qt.DisplayRole:
      if orientation == Qt.Horizontal:
        return self.names.camera[index]
      else:
        return path.splitext(self.names.image[index])[0]

  def rowCount(self, index):
    return self.view_table._shape[1]

  def columnCount(self, index):
    return self.view_table._shape[0]


class ViewModelDetections(QAbstractTableModel):
  def __init__(self, point_table, names):
    super(ViewModelDetections, self).__init__()

    self.point_table = point_table
    self.names = names

    self.detection_table = detection_tables(point_table)
    
    self.metrics = dict(detections='Detections')
    self.set_metric(0, None)
    
  @property
  def metric_labels(self):
    return list(self.metrics.values())

  @property
  def view_table(self):
    return self.detection_table.views

  def cell_color(self, detection_count):
    detection_rate = min(detection_count / self.quantiles[4], 1)

    color = Color("lime")
    color.set_luminance(0.4 + (1 - detection_rate) * 0.6 )
    return color

  def set_metric(self, metric, board, inlier_only=False):
    assert metric < len(self.metrics)
    assert board is None or board < len(self.names.board) 
    
    self.board = board
    board_table = self.detection_table.views.sum(axis=2)
    if board is not None:
      board_table = self.detection_table.views[:, :, board]

    self.quantiles = np.quantile(board_table, [0, 0.25, 0.5, 0.75, 1.0]) 
    self.modelReset.emit()
   

  def get_count(self, camera, frame):
    if self.board is None:
      return self.view_table[camera, frame].sum()
    else:
      return self.view_table[camera, frame, self.board]
 

  def data(self, index, role):
    count = self.get_count(index.column(), index.row())
    if role == Qt.DisplayRole:
      return f"{count}"

    if role == Qt.BackgroundRole:
      rgb = np.array(self.cell_color(count).get_rgb()) * 255
      return QBrush(QColor(*rgb))

  def headerData(self, index, orientation, role):
    if role == Qt.DisplayRole:
      if orientation == Qt.Horizontal:
        return self.names.camera[index]
      else:
        return path.splitext(self.names.image[index])[0]

  def rowCount(self, index):
    return self.view_table.shape[1]

  def columnCount(self, index):
    return self.view_table.shape[0]
