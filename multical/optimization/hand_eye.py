from cached_property import cached_property
import numpy as np
from structs.numpy import table
from structs.struct import subset
from .calibration import Calibration
from multical.transform.hand_eye import hand_eye_robot_world

from multical.motion import HandEye
from multical.transform import matrix
from multical import tables

class HandEyeCalibration:
  def __init__(self, calib : Calibration, gripper_wrt_base):
    
    self.valid = calib.motion.frame_poses.valid

    self.world_wrt_camera = calib.motion.frame_poses
    self.gripper_wrt_base = table(poses=gripper_wrt_base, valid=self.valid)

    base_wrt_world, gripper_wrt_camera, err = hand_eye_robot_world(
      self.world_wrt_camera.poses[self.valid], self.base_wrt_gripper.poses[self.valid])

    hand_eye_model = HandEye(self.base_wrt_gripper, np.linalg.inv(base_wrt_world), gripper_wrt_camera)
    self.calib = calib.copy(motion=hand_eye_model).enable(camera_poses=False, cameras=False)

  @cached_property
  def base_wrt_gripper(self):
    return tables.inverse(self.gripper_wrt_base)
  
  @property
  def gripper_wrt_camera(self):
    return self.model.gripper_wrt_camera

  @property
  def model(self) -> HandEye:
    return self.calib.motion

  @property
  def base_wrt_world(self):
    return np.linalg.inv(self.model.world_wrt_base)

  @cached_property
  def valid(self):
    return self.base_gripper_table.valid

  def report_error(self, name=""):
    self.calib.report(name)
    t1 = matrix.transform(self.base_wrt_world, self.world_wrt_camera.poses) 
    t2 = matrix.transform(self.base_wrt_gripper.poses, self.gripper_wrt_camera)
    matrix.report_pose_errors(t1[self.valid], t2[self.valid], name)
      

  def bundle_adjust(self):
    return self.copy(
      calib = self.calib.bundle_adjust())
  
  @cached_property
  def cameras_wrt_gripper(self):
    def with_master(k):
      hand_eye : HandEye = self.calib.with_master(k).motion
      return np.linalg.inv(hand_eye.gripper_wrt_camera)
    return {k:with_master(k) for k in self.calib.cameras.names}
    
  def __getstate__(self):
    attrs = ['gripper_wrt_base', 'calib']
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy object and change some attribute (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return self.__class__(**d)
  