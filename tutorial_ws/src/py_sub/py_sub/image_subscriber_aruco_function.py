# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import rclpy
from rclpy.node import Node
from datetime import datetime

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import copy
import argparse

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


class MinimalImageArucoSubscriber(Node):
    def __init__(self, aruco_type, intrinsics, distortion):
        super().__init__('minimal_image_aruco_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'web_cam',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cvbr = CvBridge()
        self.i = 0

        self.aruco_type_ = aruco_type
        self.aruco_dict_ = cv2.aruco.Dictionary_get(self.aruco_type_)
        self.parameters_ = cv2.aruco.DetectorParameters_create()
        self.K_ = intrinsics
        self.D_ = distortion

    def listener_callback(self, msg):
        self.get_logger().info('I received %d Image messages with frame id %s with width %d and height %d.'
                               % (self.i, msg.header.frame_id, msg.width, msg.height))
        sender_timestamp_msg = msg.header.stamp
        self.get_logger().info('The message is received at: %s' % datetime.utcfromtimestamp(sender_timestamp_msg.sec).isoformat())

        # extract the image frame from the Image msg
        frame = self.cvbr.imgmsg_to_cv2(msg)
        result_frame = copy.deepcopy(frame)
        # perform aruco marker detection
        result_frame = self.pose_estimator(result_frame)
        self.i += 1
        cv2.imshow("My webcam", result_frame)
        cv2.waitKey(1)
        
    def pose_estimator(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # detect the markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict_, parameters=self.parameters_)
        self.get_logger().info("Detected %d markers" % len(corners))
        # If markers are detected
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.097, self.K_, self.D_)
                # Draw Axis
                cv2.aruco.drawAxis(frame, self.K_, self.D_, rvec, tvec, 0.05)

            # Draw square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return frame


def main(argv=sys.argv[1:]):
    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--calibration', required=True, help="Path to the calibration data (yaml file)")
    ap.add_argument('-t', '--type', type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")

    # to avoid ros arguments
    args = ap.parse_args(rclpy.utilities.remove_ros_args(args=argv))
    args = vars(args)

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_type = ARUCO_DICT[args["type"]]
    calibration_path = args["calibration"]
    intrinsics, distortion = parse_calibration(calibration_path)

    # initialise the node
    rclpy.init()
    minimal_image_aruco_subscriber = MinimalImageArucoSubscriber(aruco_type, intrinsics, distortion)
    rclpy.spin(minimal_image_aruco_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_image_aruco_subscriber.destroy_node()
    rclpy.shutdown()


def parse_calibration(filepath):
    with open(filepath) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        intrinsics = np.asarray(data['camera_matrix']['data']).reshape((3, 3))
        distortion = np.asarray(data['distortion_coefficients']['data'])
    return intrinsics, distortion


if __name__ == '__main__':
    main()
