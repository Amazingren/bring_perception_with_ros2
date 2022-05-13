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

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class MinimalImagePublisher(Node):

    def __init__(self):
        super().__init__('minimal_image_publisher')
        self.publisher_ = self.create_publisher(Image, 'web_cam', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        
        self.videocap = cv2.VideoCapture(0)
        self.cvbridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.videocap.read()
        
        if ret == True:
            image_msg = self.cvbridge.cv2_to_imgmsg(frame)
            image_msg.header.frame_id = "web_cam"
            image_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(image_msg)
       
        self.get_logger().info('Publishing %d frame.' % self.i)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_image_publisher = MinimalImagePublisher()

    rclpy.spin(minimal_image_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_image_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
