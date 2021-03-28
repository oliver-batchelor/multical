# multical

Multi-camera calibration using one or more calibration patterns. 

![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/image_view.png)
![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/3d_view.png)

## Install

The software here is presented as a library installable from PyPi `pip install multical`, and installs a script of the same name `multical`.


## Running multical application

The script named `multical` is the primary entry point to run the calibration application. It is a wrapper around several subscripts, namely:
```
usage: multical [-h] {calibrate,check_boards,show_result} ...
```

For command line parameters, check sub-command help e.g. `multical calibrate --help`

### Input formats


The default method is to have folders named separately for each camera, with images having corresponding filenames. For example:
```
  - cam1
    - image01.jpg
    - image02.jpg
  - cam2
    - image01.jpg
    - image02.jpg
```    

Camera names and image directories can be also specified by manually specifying camera names, and optionally specifying a pattern for the directory structure. 

`multical calibrate --camera_pattern '{camera}\extrinsic' --cameras cam1,cam2,cam3` 

Where `{camera}` is replaced by the camera names 

```
  - cam1
    - intrinsic
       - image01.jpg
       - image02.jpg
    - extrinsic
       - image01.jpg
       - image02.jpg    
  - cam2
    - intrinsic
       - image01.jpg
       - image02.jpg
    - extrinsic
       - image01.jpg
       - image02.jpg    
    -cam3
    ...
```    

A fixed number of images will be chosen for initial intrinsic calibration (increase for more accuracy at expense of time with `--intrinsic_images`).

### Outputs

The outputs from `multical calibrate` are written to the `--output_path` which by default is the image path unless specified. 
* `calibration.json` - camera summary, intrinsic parameters, relative camera poses and camera rig poses
* `calibration.log`  - log file of the calibration history
* `detections.pkl`   - cached calibration pattern detections, making repeated calibrations much faster
* `workspace.pkl`    - serialized workspace containing all the details for visualization, resuming calibration etc.

### Calibration targets

Calibration targets supported are currently, charuco boards and aprilgrid boards (as used by Kalibr). Targets are configured by a configuration file with `--board` and examples can be found in the source tree: [example_boards](https://github.com/saulzar/multical/tree/master/example_boards). 

It is a good idea to check your expectation against the configuration specified using an image before calibration `multical check_boards --detect my_image.jpeg`, 

### Visualization of output

To install the libraries needed for running visualization (qtpy, pyvista-qt principly) install the `interactive` option, `pip install multical[interactive]` - these may be installed separately depending on your preference (for example with conda).

Visualization can be run by specifying the output workspace state:
`multical show_result /path/to/calibration/workspace.ws`

## Library structure


Multical provides a convenient highlevel interface found in `multical.workspace` which contains most typical useage, from finding images, loading images, initial single camera calibration, board pose extraction, pose initialisation, and finally bundle adjustment optimization and data export.

It is also the best documentation for how to use lower level library features.


## Credits

Multical derives much inspiration from the [CALICO](https://github.com/amy-tabb/calico) application, implementing largely the algorithm as presented in the paper "Calibration of Asynchronous Camera Networks: CALICO.".

A number abstractions and ideas found in [Anipose lib](https://github.com/lambdaloop/aniposelib) have been very useful, and expanded upon. Small snippets of code have been used around initialisation of relative poses which proved more robust in most cases than the least-squares method used in CALICO.

As with aniposelib, the scipy nonlinear optimizer [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) forms the basis for the bundle adjustment algorithm in this work. 

OpenCV provides many useful algorithms used heavily here, for detecting calibration boards, initialization of camera parameters and camera lens distortion models.


## Author

Oliver Batchelor
oliver.batchelor@canterbury.ac.nz
