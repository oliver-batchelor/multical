
import qtpy.QtWidgets as QtWidgets
from qtpy.QtCore import Qt

from  pyvistaqt import QtInteractor
import pyvista as pv


from .vtk_tools import *


class Viewer3D(QtWidgets.QWidget):
  def __init__(self, parent):
    super().__init__(parent)

    layout = QtWidgets.QHBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    
    self.plotter = QtInteractor(self, multi_samples=8, line_smoothing=True, point_smoothing=True)
    layout.addWidget(self.plotter.interactor)

    self.plotter.enable_anti_aliasing()
    
    for key in ["Up", "Down"]:
      self.plotter.clear_events_for_key(key)

    self.clear()    
    self.setLayout(layout)


  def clear(self):
    self.plotter.camera_set = False
    self.actors = []
    self.plotter.clear()


  def fix_camera(self):
    self.plotter.camera_set = True    

  def enable(self, enabled):
    if enabled:
      self.plotter.enable()
      self.plotter.interactor.setHidden(False)
    else:
      self.plotter.disable()
      self.plotter.interactor.setHidden(True)


  def add_mesh(self, mesh, transform=None, **kwargs):
      actor = self.plotter.add_mesh(mesh, **kwargs)

      if transform is not None: 
        actor.SetUserTransform(vtk_transform(transform))

      self.actors.append(actor)
      return actor


  def set_point_size(self, size):
    for actor in self.actors:
        actor.GetProperty().SetPointSize(size)
    self.update()

  def set_line_size(self, size):
    for actor in self.actors:
        actor.GetProperty().SetLineWidth(size)
    self.update()

  def update(self):
    self.plotter.interactor.update()


  def current_viewport(self):
    vp = save_viewport(self.plotter)
    return vp

  def set_viewport(self, viewport):
    set_viewport(self.plotter, viewport)
    self.update()
    

  def camera_viewport(self, camera, pose):
    return camera_viewport(camera.intrinsic, pose, self.plotter.window_size)
