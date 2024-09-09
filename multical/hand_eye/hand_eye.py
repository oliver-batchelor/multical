import os.path

from structs.numpy import shape_info, struct, Table, shape
import numpy as np
from multical.transform import matrix
import json
import pickle
import gc
from .helper import *
from multical.io.logging import debug, info
import cv2
class HandEye():
  def __init__(self, pose_table, cam_name, image_path):
      assert isinstance(pose_table, Table)
      self.cam_names = cam_name
      self.pose_table = pose_table
      self.image_path = image_path
      self.num_cameras = self.pose_table._prefix[0]
      self.num_images = self.pose_table._prefix[1]
      self.num_boards = self.pose_table._prefix[2]
      self.viewed_boards = self.check_viewed_boards()
      self.camera_poses = {}
      self.camera_groups = {}
      self.reference_camera = None
      self.cam_init = {}
      self.handeye_df = []

  def check_viewed_boards(self):
      limit_board_image = 6
      boards = {}
      for idx, cam in enumerate(self.cam_names):
          boards[cam] = [b for b in range(0, self.num_boards) if
                           self.pose_table.valid[idx][:, b].sum() > limit_board_image]
      return boards
  def initialise_camera_poses(self):
      max_density = 0
      max_group = 0
      for idm, master_cam in enumerate(self.cam_names):
          temp_density = 0
          temp_group = 0
          self.camera_poses[master_cam] = {}
          self.camera_groups[master_cam] = {}
          self.camera_poses[master_cam][master_cam] = np.eye(4).tolist()
          for ids, slave_cam in enumerate(self.cam_names):
              if slave_cam != master_cam:
                  cs_wrto_cm = []
                  for boardM in self.viewed_boards[master_cam]:
                      for boardS in self.viewed_boards[slave_cam]:
                          slaveCam_wrt_masterCam, image_ids = self.master_slave_pair(idm, ids, boardM, boardS)
                          if slaveCam_wrt_masterCam is not None:
                              self.handeye_df.append({"master_cam": master_cam, "slave_cam":slave_cam, "boardM": boardM, "boardS":boardS,
                                                      "image_ids": image_ids, "slaveCam_wrt_masterCam":slaveCam_wrt_masterCam})
                              cs_wrto_cm.append(slaveCam_wrt_masterCam)
                  self.camera_groups[master_cam][slave_cam] = [g.tolist() for g in cs_wrto_cm]
                  density, self.camera_poses[master_cam][slave_cam] = probabilistic_guess(cs_wrto_cm)
                  temp_density += density
                  temp_group += len(cs_wrto_cm)
          # for choosing reference camera
          if (temp_density > max_density and temp_group > max_group):
              max_density = temp_density
              max_group = temp_group
              self.reference_camera = master_cam

      self.camera_poses['Reference_camera'] = self.reference_camera
      # Export all groups and group mean
      with open(os.path.join(self.image_path, 'camera_groups.pkl'), 'wb') as f:
          pickle.dump(self.camera_groups, f)
      del self.camera_groups
      gc.collect()

      with open(os.path.join(self.image_path,'initial_guess.json'), 'w') as fp:
          json.dump(self.camera_poses, fp)

      info(f"Reference camera {self.reference_camera} with density {max_density} and number of groups {max_group}")
      init0 = relative_to_cam(self.cam_names[0], self.camera_poses[self.reference_camera])
      # for Multical camera pose, we need masterCam_wrto_slaveCam
      for k in self.cam_names:
          init0[k] = np.linalg.inv(init0[k])
      self.cam_init = init0


  def master_slave_pair(self, master_cam, slave_cam, boardM, boardS):
      """
      master_cam : Reference camera
      slave_cam : Cameras except master_cam
      boardM : Boards that are viewed by the master_cam
      boardS : Boards that are viewed by a slave_cam
      """
      table_m = self.pose_table._index_select(master_cam, axis=0)._index_select(boardM, axis=1)
      table_s = self.pose_table._index_select(slave_cam, axis=0)._index_select(boardS, axis=1)

      valid_images = table_m.valid & table_s.valid
      image_ids = np.flatnonzero(valid_images)
      if len(image_ids) < 3:
          return None, None
      master_poses = [np.linalg.inv(p) for p in table_m.poses[image_ids]]
      slave_poses = [np.linalg.inv(p) for p in table_s.poses[image_ids]]
      master_R = [matrix.split(p)[0] for p in master_poses]
      master_t = [matrix.split(p)[1] for p in master_poses]
      slave_R = [matrix.split(p)[0] for p in slave_poses]
      slave_t = [matrix.split(p)[1] for p in slave_poses]

      (slaveCam_wrt_masterCam, slaveB_wrt_masterB) = self.hand_eye_robot_world(master_R, master_t, slave_R, slave_t)
      if slaveB_wrt_masterB is not None:
        return slaveCam_wrt_masterCam, image_ids
      else:
        return None, None

  @staticmethod
  def hand_eye_robot_world(cam_world_R, cam_world_t, base_gripper_R, base_gripper_t):
      try:
          base_cam_r, base_cam_t, gripper_world_r, gripper_world_t = \
            cv2.calibrateRobotWorldHandEye(cam_world_R, cam_world_t, base_gripper_R, base_gripper_t, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

          base_wrt_cam = matrix.join(base_cam_r, base_cam_t.reshape(-1))
          gripper_wrt_world = matrix.join(gripper_world_r, gripper_world_t.reshape(-1))
          return base_wrt_cam, gripper_wrt_world
      except:
          print('normalizeRotation error')
          ## Add Rotation Normalization check
          return None, None



