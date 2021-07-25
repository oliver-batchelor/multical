from multical.io.report import report_pose_errors
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
  def __init__(self, calib : Calibration, gripper_wrt_base, world_wrt_camera):
    assert isinstance(calib.motion, HandEye)

    self.gripper_wrt_base = gripper_wrt_base
    self.world_wrt_camera = world_wrt_camera
    
    self.calib = calib

  @staticmethod 
  def initialise(calib, gripper_wrt_base):
      world_wrt_camera = calib.motion.frame_poses.poses
      valid = calib.motion.frame_poses.valid

      base_wrt_gripper = np.linalg.inv(gripper_wrt_base)
        # Initialize with hand-eye motion model
      base_wrt_world, gripper_wrt_camera, err = hand_eye_robot_world(
        world_wrt_camera[valid], base_wrt_gripper[valid])

      hand_eye_model = HandEye(table(poses=base_wrt_gripper, valid=valid), 
        np.linalg.inv(base_wrt_world), gripper_wrt_camera)

      calib = calib.copy(motion=hand_eye_model).enable(camera_poses=False, cameras=False)
      return HandEyeCalibration(calib, gripper_wrt_base, world_wrt_camera)

  @property 
  def valid(self):
    return self.calib.motion.valid


  @cached_property
  def gripper_wtr_base_table(self):
    return table(poses=self.gripper_wrt_base, valid=self.valid)

  @cached_property
  def base_wrt_gripper_table(self):
    return tables.inverse(self.gripper_wtr_base_table)
  
  @property
  def gripper_wrt_camera(self):
    return self.model.gripper_wrt_camera

  @property
  def model(self) -> HandEye:
    return self.calib.motion

  @property
  def base_wrt_world(self):
    return np.linalg.inv(self.model.world_wrt_base)


  def report_error(self, name=""):   
    self.calib.report(name)
 
    report_pose_errors(
      self.calib.motion.frame_poses.poses[self.valid], 
      self.world_wrt_camera[self.valid], 
      name)
      

  def bundle_adjust(self):
    return self.copy(
      calib = self.calib.bundle_adjust())
  
  def adjust_outliers(self, **kwargs):
    return self.copy(
      calib=self.calib.adjust_outliers(**kwargs))


  @cached_property
  def cameras_wrt_gripper(self):
    def with_master(k):
      hand_eye : HandEye = self.calib.with_master(k).motion
      return np.linalg.inv(hand_eye.gripper_wrt_camera)
    return {k:with_master(k) for k in self.calib.cameras.names}
    
  def __getstate__(self):
    attrs = ['gripper_wrt_base', 'world_wrt_camera', 'calib']
    return subset(self.__dict__, attrs)

  def copy(self, **k):
    """Copy object and change some attribute (no mutation)"""
    d = self.__getstate__()
    d.update(k)
    return self.__class__(**d)
  