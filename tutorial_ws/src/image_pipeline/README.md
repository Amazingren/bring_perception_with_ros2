image_pipeline
==============

[![](https://github.com/ros-perception/image_pipeline/workflows/Basic%20Build%20Workflow/badge.svg?branch=ros2)](https://github.com/ros-perception/image_pipeline/actions)

This package fills the gap between getting raw images from a camera driver and higher-level vision processing.

For more information on this metapackage and underlying packages, please see [the ROS wiki entry](http://wiki.ros.org/image_pipeline).

This version is based on the clone of ros2 branch with some fixes:
* fix the *handle_monocular* function in *camera_calibrator* with the line 'checkerboard_flags=self._checkerboard_flags' in *camera_calibrator.py*
* fix the 'No fisheye_flags' error by adding 'fisheye_flags = 0' at the end of the '_init_()' function under CalibrationNode class in *camera_calibrator.py*
* save the calibration data into the HOME/tutorial_ws instead of /tmp folder by modifying the 'do_save()' function in *calibrator.py*
