import math
from os import path
from multical import tables

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QAbstractTableModel
import numpy as np

from structs.struct import struct
from structs.numpy import shape, Table



def masked_quantile(error, mask, quantiles, axis=None):
  error = error.copy()
  error[~mask] = math.nan

  return np.nanquantile(error, quantiles, axis=axis)

def reprojection_statistics(error, valid, inlier, axis=None):
  n = valid.sum(axis=axis)
  mse = (error * error).sum(axis=axis) / n

  outliers = (valid & ~inlier).sum(axis=axis)
  quantiles = masked_quantile(error, valid, [0, 0.25, 0.5, 0.75, 1.0], axis=axis)

  return Table.create(detected=n, outliers=outliers, mse=mse, rms=np.sqrt(mse),
    quantiles = np.moveaxis(quantiles, 0, -1))

def reprojection_tables(calib, inlier_only=False):
  error, valid = tables.reprojection_error(calib.projected, 
    calib.inliers if inlier_only else calib.point_table)

  f = lambda axis: reprojection_statistics(error, valid, calib.inliers, axis=axis)
  axes =  struct(overall = None, views = 2, cameras = (1, 2), frames = (0, 2))
  return axes._map(f)




class ViewModel(QAbstractTableModel):
    def __init__(self, calib, camera_names, image_names):
        super(ViewModel, self).__init__()
        self.reprojection_table = reprojection_tables(calib)

        self.camera_names = camera_names
        self.image_names = image_names

    @property
    def view_table(self):
      return self.reprojection_table.views

    def data(self, index, role):
        view_stats = self.view_table._index[index.column(), index.row()]

        if role == Qt.DisplayRole:
          return f"{view_stats.detected}"

    def headerData(self, index, orientation, role):
      if role == Qt.DisplayRole:
        if orientation == Qt.Horizontal:
          return self.camera_names[index]
        else:
          return path.splitext(self.image_names[index])[0]

    def rowCount(self, index):
        return self.view_table._shape[1]

    def columnCount(self, index):
        return self.view_table._shape[0]