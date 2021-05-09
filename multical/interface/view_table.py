from collections import OrderedDict
import math
from numbers import Integral
from os import path

from qtpy.QtGui import QBrush, QColor
from multical import tables

from qtpy.QtCore import Qt
from qtpy.QtCore import QAbstractTableModel
import numpy as np

from structs.struct import struct, split_dict
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


sum_axes = struct(overall=None, views=(2, 3), board_views=(3,),
  boards=(0, 1, 3), cameras=(1, 2, 3), frames=(0, 2, 3))

def reprojection_tables(calib, inlier_only=False):
  point_table = calib.point_table
  if inlier_only:
    point_table = point_table._extend(valid=calib.inliers)

  error, valid = tables.reprojection_error(calib.projected, point_table)

  def f(axis): return reprojection_statistics(
      error, valid, calib.inliers, axis=axis)
  return sum_axes._map(f)


def get_view_metric(reprojection_table, metric, board=None):
  views = (reprojection_table.views if board is None 
    else reprojection_table.board_views._index[:, :, board])

  return views[metric]


def detection_tables(point_table):
  valid = point_table.valid
  def f(axis): return np.sum(valid, axis=axis)
  return sum_axes._map(f)


def hsl_color(name):
  hsl = Color(name).get_hsl()
  return np.array(hsl)

def lerp(t, x, y):
  return t * x + (1 - t) * y

def lerp_table(t, x, y):
  assert x.shape == y.shape
  et = np.expand_dims(t, tuple(t.ndim + np.arange(x.ndim)))
  ex = np.expand_dims(x, tuple(np.arange(t.ndim)))
  ey = np.expand_dims(y, tuple(np.arange(t.ndim)))

  return lerp(et, ex, ey)



class ViewModelCalibrated(QAbstractTableModel):
  def __init__(self, calib, names):
    super(ViewModelCalibrated, self).__init__()

    self.calib = calib
    self.reproj = reprojection_tables(calib)
    self.reproj_inl = reprojection_tables(calib, inlier_only=True)

    self.names = names

    self.metric_types = OrderedDict(
        detected='Detected',
        median='Median',
        upper_q='Upper quartile',
        max='Maximum',
        rms='Root Mean Square',
        mse='Mean Square Error'
    )

    self.set_metric(0, None)

  @property
  def metric_labels(self):
    return list(self.metric_types.values())


  def make_cell_color_table(self, board):
    detections, inliers = self.get_metric_tables('detected', board)
    upper_q = np.quantile(detections, 0.75)

    outlier_rate = (detections - inliers) / np.maximum(1, detections)
    outlier_t = np.clip(outlier_rate * 5, 0.0, 1.0)
    colors = lerp_table(outlier_t, hsl_color("red"), hsl_color("green"))

    detection_rate = np.clip(detections / np.maximum(upper_q, 1), 0.0, 1.0)

    colors[..., 2] = np.clip(1 - detection_rate + 0.25, 0.5, 1.0)
    return colors 

  def get_metric_tables(self, metric, board=None):
    return get_view_metric(self.reproj, metric, board), get_view_metric(self.reproj_inl, metric, board)

  def set_metric(self, index, board=None):
    assert isinstance(index, int) and index < len(self.metric_types)
    keys, _ = split_dict(self.metric_types)
    self.metric = keys[index]

    self.view_table, self.view_table_inl = self.get_metric_tables(self.metric, board)
    self.cell_color_table = self.make_cell_color_table(board)
    self.modelReset.emit()

  def data(self, index, role):
    def format_nan(x):
      return "" if math.isnan(x) else f"{x:.2f}"

    all = self.view_table[index.column(), index.row()]
    inlier = self.view_table_inl[index.column(), index.row()]

    if role == Qt.DisplayRole:
      return f"{inlier} ({all})" if isinstance(inlier, Integral)\
        else f"{format_nan(inlier)} ({format_nan(all)})" 

    if role == Qt.BackgroundRole:
      hsl = self.cell_color_table[index.column(), index.row()]
      return QBrush(QColor.fromHslF(*hsl))

  def headerData(self, index, orientation, role):
    if role == Qt.DisplayRole:
      if orientation == Qt.Horizontal:
        return self.names.camera[index]
      else:
        return path.splitext(self.names.image[index])[0]

  def rowCount(self, index):
    return len(self.names.image)

  def columnCount(self, index):
    return len(self.names.camera)


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
    detection_rate = np.clip(detection_count / self.quantiles[4], 0, 1)
    color = hsl_color("lime")
    color[2] = 0.4 + (1 - detection_rate) * 0.6
    return color 

  def set_metric(self, metric, board):
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
      hsl = self.cell_color(count) 
      return QBrush(QColor.fromHslF(*hsl))

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
