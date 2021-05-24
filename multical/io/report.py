import numpy as np
from .logging import info

from multical.transform import matrix

def report_errors(name, errs):
  rms = np.sqrt(np.square(errs).mean())
  quantiles = np.quantile(errs, [0, 0.25, 0.5, 0.75, 1.0])
  info(f"{name} - RMS: {rms:.4f} quantiles: {quantiles}")

def report_pose_errors(p1, p2, k = ""):
  err = matrix.pose_errors(p1, p2)
  info(f"{k} pose errors:")
  report_errors("translation", err.translation)
  report_errors("angle(deg)", err.rotation_deg)
  report_errors("frobius", err.frobius)
