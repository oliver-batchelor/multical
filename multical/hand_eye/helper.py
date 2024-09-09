import numpy as np
from multical.tables import *
from multical.transform import *
from scipy import stats
def probabilistic_guess(transformations_list):
    """
    Selects the best translation vector (position of camera) from all Master-Slave groups
    """
    assert isinstance(transformations_list, list) and transformations_list[0].shape == (4,4)
    x = [rtvec.as_rtvec(p)[3:][0] for p in transformations_list]
    y = [rtvec.as_rtvec(p)[3:][1] for p in transformations_list]
    z = [rtvec.as_rtvec(p)[3:][2] for p in transformations_list]

    xyz = np.vstack([x, y, z])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    max_idx = np.argmax(density)
    return density[max_idx], transformations_list[max_idx].tolist()

def relative_to_cam(new_ref, camera_poses):
    new_pose = np.linalg.inv(camera_poses[new_ref])
    for k, v in camera_poses.items():
        camera_poses[k] = new_pose @ camera_poses[k]
    return camera_poses
