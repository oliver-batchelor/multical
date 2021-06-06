import cv2
import numpy as np
from . import matrix

from packaging import version


def hand_eye_robot_world_t(camera_wrt_world, gripper_wrt_base):
  """
    A version of hand_eye_robot_world: Using the opposite transform convention:
    gripper_wrt_base represents poses of the gripper in the base frame, and
    camera_wrt_world represents poses of the camera in a world frame
  """
  base_wrt_world, gripper_wrt_camera, err = hand_eye_robot_world(
      np.linalg.inv(camera_wrt_world), np.linalg.inv(gripper_wrt_base))
  return np.linalg.inv(base_wrt_world), np.linalg.inv(gripper_wrt_camera), err



def hand_eye_robot_world(world_wrt_camera, base_wrt_gripper):
  """
    Solve the robot-world hand-eye problem AX=ZB
    In particular more commonly solve for known world_camera and base_gripper
    world_wrt_camera @ base_wrt_gripper =  gripper_wrt_camera @ base_wrt_gripper

    Note: Uses the data-centric convention where (like OpenCV) 
    describing coordinate change of points
  """

  # Robot world hand eye only added in OpenCV 4.5
  if version.parse(cv2.__version__) < version.parse('4.5.0'):
    return hand_eye(world_wrt_camera, base_wrt_gripper)

  assert world_wrt_camera.shape[0] == base_wrt_gripper.shape[0]

  world_camera_r, world_camera_t = matrix.split(world_wrt_camera)
  base_gripper_r, base_gripper_t = matrix.split(base_wrt_gripper)

  base_world_r, base_world_t, gripper_cam_r, gripper_cam_t =\
    cv2.calibrateRobotWorldHandEye(
      world_camera_r, world_camera_t, 
      base_gripper_r, base_gripper_t)

  base_wrt_world = matrix.join(base_world_r, base_world_t.reshape(-1))
  gripper_wrt_camera = matrix.join(gripper_cam_r, gripper_cam_t.reshape(-1))

  err = matrix.transform(base_wrt_world, world_wrt_camera) - matrix.transform(base_wrt_gripper, gripper_wrt_camera)
  

  return base_wrt_world, gripper_wrt_camera, np.linalg.norm(err, axis=(1, 2))


def hand_eye_t(camera_wrt_world, gripper_wrt_base):
  """
    A version of hand_eye_robot_world: Using the opposite transform convention:
    base_gripper represents poses of the gripper in the base frame, and
    world_camera represents poses of the camera in a world frame
  """
  gripper_camera = hand_eye(
      np.linalg.inv(camera_wrt_world), np.linalg.inv(gripper_wrt_base))
  return np.linalg.inv(gripper_camera)

def hand_eye(world_wrt_camera, base_wrt_gripper):
  """
    Solve the hand-eye problem AX=XB
    See cv2.calibrateHandEye for details. 
    
    Inputs changed to be consistent with hand_eye_robot_world
    compared to cv2.calibrateHandEye. 
    
    Less accurate than hand_eye_robot_world, used as fallback in OpenCV < 4.5

    Note: Uses the data-centric convention where world_camera describes the 
    transform which sends a *point* in the world frame to the same point in camera frame.
  """

  assert world_wrt_camera.shape[0] == base_wrt_gripper.shape[0]

  world_camera_r, world_camera_t = matrix.split(world_wrt_camera)
  base_gripper_r, base_gripper_t = matrix.split(np.linalg.inv(base_wrt_gripper))

  camera_gripper_r, camera_gripper_t =  cv2.calibrateHandEye(
    base_gripper_r, base_gripper_t,
    world_camera_r, world_camera_t)

  camera_wrt_gripper = matrix.join(camera_gripper_r, camera_gripper_t.reshape(-1))
  gripper_wrt_camera = np.linalg.inv(camera_wrt_gripper)

  base_wrt_world = matrix.mean_robust(matrix.transform(
    base_wrt_gripper, gripper_wrt_camera, np.linalg.inv(world_wrt_camera)))

  err = matrix.transform(base_wrt_world, world_wrt_camera) - matrix.transform(base_wrt_gripper, gripper_wrt_camera)
  return base_wrt_world, gripper_wrt_camera, np.linalg.norm(err, axis=(1, 2))
