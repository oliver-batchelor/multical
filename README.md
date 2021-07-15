# multical 


Multi-camera calibration using one or more calibration patterns. 

![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/image_view.png)
![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/3d_view.png)

[changelog](https://github.com/saulzar/multical/tree/master/example_boards)

## Install

The software here is presented as a library installable from PyPi `pip install multical`, and installs a script of the same name `multical`.


## Running multical application

The script named `multical` is the primary entry point to run the calibration application. It is a wrapper around several subscripts, namely:
```
usage: multical [-h] {calibrate,intrinsic,boards,show} ...
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

`multical calibrate --camera_pattern '{camera}/extrinsic' --cameras cam1,cam2,cam3` 

By default the current directory is searched, it can be specified with `--image_path`.

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

A fixed number of images will be chosen for initial intrinsic calibration (increase for more accuracy at expense of time with `--limit_intrinsic`).

### Outputs

The outputs from `multical calibrate` are written to the `--output_path` which by default is the image path unless specified. The name `calibration` (default) is specified by `--name`.

* `calibration.json` - camera summary, intrinsic parameters, relative camera poses and camera rig poses

* `calibration.log`  - log file of the calibration history
* `calibration.detections.pkl`   - cached calibration pattern detections, making repeated calibrations much faster
* `calibration.pkl`    - serialized workspace containing all the details for visualization, resuming calibration etc.

### Calibration targets

Calibration targets supported are currently, charuco boards and aprilgrid boards (as used by Kalibr). Targets are configured by a configuration file with `--board` and examples can be found in the source tree: [example_boards](https://github.com/saulzar/multical/tree/master/example_boards). 

It is a good idea to check your expectation against the configuration specified using an image before calibration `multical boards --boards my_board.yaml --detect my_image.jpeg`, 

### Visualization of output

To install the libraries needed for running visualization (qtpy, pyvistaqt principly) install the `interactive` option, `pip install multical[interactive]` - these may be installed separately depending on your preference (for example with conda).

Visualization can be run by :
`multical vis --workspace_file calibration.pkl`



## Library structure


Multical provides a convenient highlevel interface found in `multical.workspace` which contains most typical useage, from finding images, loading images, initial single camera calibration, board pose extraction, pose initialisation, and finally bundle adjustment optimization and data export.

It is also the best documentation for how to use lower level library features.


## FAQ

### How do I make a (physical) board pattern?
Here's my workflow for making board images:

```
multical boards --boards example_boards/charuco_16x22.yaml --paper_size A2 --pixels_mm 10 --write my_images 

Using boards:
charuco_16x22 CharucoBoard {type='charuco', aruco_dict='4X4_1000', aruco_offset=0, size=(16, 22), marker_length=0.01875, square_length=0.025, aruco_params={}}
Wrote my_images/charuco_16x22.png
```

Then open up images/charuco_16x22.png in gimp and print-to-file (pdf) with the margins set to zero and the paper size set to A2. Print pdf to printer or send to print shop.

### Can multical calibrate intrinsic and extrinsic camera parameters together?

Yes - this is the default, the same images will be used for both intrinsic and extrinsic calibration. See below for how to do separate intrinsic and extrinsic calibration.

### Can multical calibrate intrinsic parameters separately?

Yes - first calibrate separate intrinsic-only calibration (with images not needing to be corresponding), this will produce a calibration per camera in `intrinsic.json`.

`multical intrinsic --input_path intrinsic_images` 

Extrinsic-only calibration can then be performed using the known intrinsic parameters with `--calibration` to specify a prior intrinsic calibration to use, combined with `--fix_intrinsic` in order to avoid adjusting those parameters.

`multical calibrate --input_path extrinsic_images --calibration intrinsic_images/intrinsic.json --fix_intrinsic`


### How can I diagnose a bad calibration?

* Check that the boards are detected as expected (see above section on Calibration targets), in particular make sure the dimensions are correct for the pattern (e.g. 8x6 vs. 6x8)

* Ensure a correct camera model is being used. Note that there are currently no fisheye camera models - feel free to add one!
* Visualize the result and check that it matches expectations - look for patterns in the errors, is it just with particular frames or cameras? Check the initialization and verify if it is a problem with detecting the pattern or if the bundle adjustment step is causing the issue.

* Ensure input images are synchronized properly (images not captured at the same time will not calibrate well), or appropriate measures are taken to keep cameras still - for example a tripod.


### How can I evaluate the accuracy of a calibration?

* Reprojection error - this evaluates how well the models can fit the data. Low reprojection error is a good calibration if sufficient quantity and variation of input images are used. If the inputs are too few, the camera models may not be constrained enough to produce a good calibration.

* Compare a calibration against a different image set, captured with the same cameras. Fix camera parameters with `--fix_intrinsic` and `--fix_camera_poses` and calibrate on the alternative image set:

`multical calibrate --input_path alternative_images --calibration calibration.json --fix_intrinsic --fix_camera_poses`

The reprojection error will be high if the camera parameters and poses don't match the alternative image set well.

## Credits

Multical derives much inspiration from the [CALICO](https://github.com/amy-tabb/calico) application, implementing largely the algorithm as presented in the paper "Calibration of Asynchronous Camera Networks: CALICO.".

A number abstractions and ideas found in [Anipose lib](https://github.com/lambdaloop/aniposelib) have been very useful, and expanded upon. Small snippets of code have been used around initialisation of relative poses which proved more robust in most cases than the least-squares method used in CALICO.

As with aniposelib, the scipy nonlinear optimizer [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) forms the basis for the bundle adjustment algorithm in this work. 

OpenCV provides many useful algorithms used heavily here, for detecting calibration boards, initialization of camera parameters and camera lens distortion models.

### TODO/Incomplete parts

* Make the buttons on the UI operational to be a complete application rather than just a visualzer
* Extend apriltags2_ethz to pass through tag dictionaries other than the default t36h11
* Make the error display grid on the right hand tab use the tolerance slider and adapt color depending on metric
* Continuous time rolling shutter camera model
* Add ability to calibrate with cameras which have no overlap (using hand-eye AX = ZB initialization method)



## Author

Oliver Batchelor
oliver.batchelor@canterbury.ac.nz
