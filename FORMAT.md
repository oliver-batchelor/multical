# Calibration file format

Example calibration json file, camera intrinsics followed by camera poses.

## Intrinsics: 
- image_size: [width, height]
- model: lens distortion model type, one of: standard|rational|thin_prism|tilted
- K: 3x3 camera matrix 
- dist: 1x{4,5,8,11,14} distortion coefficients 

## Camera poses: 
Relative poses of cameras, may either be an absolute camera pose e.g. "cam1", or a relative camera pose in the form of "cam2_to_cam1".

- R: 3x3 rotation 
- T: (3) translation


## Example file
{
  "cameras": {
    "cam1": {
      "image_size": [
        2000.00,
        1500.00
      ],
      "K": [
        [2251.52, 0.00,     985.68],
        [0.00,    2252.80,  763.00],
        [0.00,    0.00,     1.00]
      ],
      "dist": [-0.12, 0.32,-0.00, -0.00,-0.20]
    },
    "cam2": {
      "image_size": [2000.00, 1500.00],
      "K": [
        [2257.19, 0.00,    983.84],
        [0.00,    2258.64, 754.08],
        [0.00, 0.00, 1.00]
      ],
      "dist": 
        [ -0.12, 0.39, -0.00, -0.00, -0.28]
    }
  },
  "camera_poses": {
    "cam1": {
      "R": [
        [ 1.00, 0.00, 0.00 ],
        [ 0.00, 1.00, 0.00 ],
        [ 0.00, 0.00, 1.00 ]
      ],
      "T": [ 0.00, 0.00, 0.00 ]
    },
    "cam2_to_cam1": {
      "R": [
        [ 1.00,  0.00, -0.01 ],
        [ -0.00, 1.00,  0.00 ],
        [ 0.01, -0.00,  1.00 ]
      ],
      "T": [ -0.05, 0.00, 0.00]
    }
  }
}  
    
```
