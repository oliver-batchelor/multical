import numpy as np
import vtk

from structs.struct import struct

def vtk_matrix(t):
    """Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    t = np.linalg.inv(t)

    matrix = vtk.vtkMatrix4x4()
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            matrix.SetElement(i, j, t[i, j])
    return matrix

def vtk_transform(t):
  transform = vtk.vtkTransform()
  transform.SetMatrix(t.flatten())
  return transform

def save_viewport(plotter):
    viewport = plotter.camera

    return struct(
        position = viewport.GetPosition(),
        focal_point = viewport.GetFocalPoint(),
        up = viewport.GetViewUp(),
        transform = viewport.GetModelTransformMatrix(),
        center = viewport.GetWindowCenter(),
        angle = viewport.GetViewAngle()
    )   

def set_viewport(plotter, camera):
    viewport = plotter.camera

    viewport.SetPosition(*camera.position)
    viewport.SetFocalPoint(*camera.focal_point)
    viewport.SetViewUp(*camera.up)

    viewport.SetModelTransformMatrix(camera.transform)
    viewport.SetWindowCenter(*camera.center)
    viewport.SetViewAngle(camera.angle)
    
    viewport.SetClippingRange(0.01, 1000)
    plotter.reset_camera_clipping_range()


def camera_viewport(intrinsic, extrinsic, window_size):
    c = intrinsic[:2, 2]
    f = (intrinsic[0,0], intrinsic[1, 1])

    w, h = window_size
    
    angle_x = 180 / np.pi * 2.0 * np.arctan2(w / 2.0, f[0])

    return struct(
        position = (0, 0, 0),
        focal_point = (0, 0, 1),
        up = (0, -1, 0),
        transform = vtk_matrix(extrinsic),
        center = (0, 0),
        angle = angle_x
    )  