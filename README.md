# multical

Multi-camera calibration using one or more calibration patterns. 


## Install

The software here is presented as a library installable from PyPi `pip install multical`, and contains has an application of the same name `multical`.


## Running multical




### Visualization of output




## Library structure


Multical provides a convenient highlevel interface found in `multical.workspace` which 



## Credits

Multical derives much inspiration from the [CALICO](https://github.com/amy-tabb/calico) application, implementing largely the algorithm as presented in the paper "Calibration of Asynchronous Camera Networks: CALICO.".

A number abstractions and ideas found in [Anipose lib](https://github.com/lambdaloop/aniposelib) have been very useful, and expanded upon. Small snippets of code have been used around initialisation of relative poses which proved more robust in most cases than the least-squares method used in CALICO.

As with aniposelib, the scipy nonlinear optimizer [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) forms the basis for the bundle adjustment algorithm in this work. 

OpenCV provides many useful algorithms used heavily here, for detecting calibration boards, initialization of camera parameters and camera lens distortion models.


## Author

Oliver Batchelor
oliver.batchelor@canterbury.ac.nz

