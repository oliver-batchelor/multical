# Calibration file format

Example calibration file, camera intrinsics followed by extrinsics. Optionally, followed by scan information, with 'rig_poses' (if available) and 'image_sets'.
JSON format (or for long sequences numpy pickled).

This format can either describe RGB scans and RGBD scans (stereo matched or Zivid scans). For RGB scans 'image_sets' will contain just 'rgb' and for RGBD, will contain 'rgb' and 'depth'. 

## Intrinsics: 
    - image_size
    - K: 3x3 camera matrix (K)
    - dist: 1x5 distortion coefficients 

## Extrinsics: 
    Transform of camera object relative to another camera (parent), can describe a rig of cameras, or distinct stereo pairs.

    - R: 3x3 rotation 
    - T: (3) translation
    - parent: reference camera

    Note this transform describes the transformation of the *camera object* in the frame of the camera rig. 
    The OpenCV convention describes the transformation of points in the *local camera space* from the left camera to the right camera. This the inverse of the camera object transform from left camera to right camera.
       

## Stereo pairs:
    - stereo_pairs: A (named) dict of camera pairs which can be used for stereo matching.
    

## Extended file format:
    - rig_poses: A list of 4x4 matrices for extrinsics of the camera for each frame 
    - image_sets: A dict of image subsets (e.g. rgb, depth), each subset containing a list of frames with an entry for each camera
   
Scans in the extended file format can be viewed/manipulated by tools in this repository:
https://github.com/maraatech/array_stereo_reconstruction

## Example file

JSON format, numbers truncated for readibility.

```
{
    "cameras": {
        "cam1": {
            "image_size": [4000, 3000],
            "K": 
                [[4480.30, 0.0,     2049.97],
                [0.0,     4478.47, 1524.04],
                [0.0,     0.0,     1.0]],
            "dist": [[-0.117, 0.285, -0.000, 0.000,  -0.048]]
        },
        "cam2": {
            "image_size": [4000, 3000],
            "K": [[4483.51,     0.0,        2022.67],
                  [0.0,         4481.26,    1536.34],
                  [0.0,         0.0,        1.0]],
            "dist": [[-0.117, 0.285, -0.000, 0.000,  -0.048]]
        }
    },
    "extrinsics": {
        # Right camera centre is 5cm to the right of the left camera and 
        "cam2": {
            "R": [[0.999,  0.001,  0.000],
                  [-0.001, 0.999,  0.002],
                  [-0.003, -0.032, 0.987]],
            "T": [0.503, 0.003, 0.005],        
            "parent": "cam1"
        }
    },
    "stereo_pairs": {
        "stereo1": ["cam1","cam2"]
    }
    
    # Extended scan format only below here
    "rig_poses": [
        [
          0.072, -0.04,  0.99, -1.57,
          -0.05,  0.99,  0.05, -0.79,
          -0.99, -0.05,  0.06,  0.44,
          0.0,    0.0,   0.0,   1.0
        ],
        [
          0.071, -0.04,  0.99, -1.58,
          -0.05,  0.99,  0.05, -0.72,
          -0.99, -0.056, 0.068, 0.43,
          0.0,    0.0,   0.0,   1.0
        ],
    ]
    
    "image_sets": {
         "rgb": [
              {
                "cam1": "rgb/image0000_cam1.jpg",
                "cam2": "rgb/image0000_cam2.jpg",
              },
              {
                "cam1": "rgb/image0005_cam1.jpg",
                "cam2": "rgb/image0005_cam2.jpg",
              }
          ],

          "depth": [
              {
                "cam1": "depth/image0000_cam1.png",
                "cam2": "depth/image0000_cam2.png",
              },
              {
                "cam1": "depth/image0005_cam1.png",
                "cam2": "depth/image0005_cam2.png",
              },
          ]
      }
}
```
