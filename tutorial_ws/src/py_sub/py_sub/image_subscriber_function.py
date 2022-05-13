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
import os
import rclpy
from rclpy.node import Node
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class MinimalImageSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'web_cam',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cvbr = CvBridge()
        self.i = 0
        self.save_path = None
        # If a data path is provided, save the frames
        if len(sys.argv)>=2:
            self.get_logger().info('There are %d args' % len(sys.argv) )
            self.save_path = sys.argv[1]
            # check if the folder exists. If not, create one
            if not os.path.exists(self.save_path):
                self.get_logger().info('Create the folder for saving the video frames')
                os.makedirs(self.save_path)
    def listener_callback(self, msg):
        self.get_logger().info('I received %d Image messages with frame id %s with width %d and height %d.'
                               % (self.i, msg.header.frame_id, msg.width, msg.height))
        sender_timestamp_msg = msg.header.stamp
        print('The message is received at: %s' % datetime.utcfromtimestamp(sender_timestamp_msg.sec).isoformat())

        sender_timestamp = rclpy.time.Time.from_msg(sender_timestamp_msg)
        receiver_timestamp = self.get_clock().now()
        delay = receiver_timestamp - sender_timestamp  # obtain Duration
        print("The delay at the receiver in seconds is %.9f s." % (delay.nanoseconds / 1e9))

        # extract the image frame from the Image msg
        frame = self.cvbr.imgmsg_to_cv2(msg)
        # save/visualise the image frame
        if self.save_path is not None:
            file_name = os.path.join(self.save_path, 'frame%06d.png' % self.i)
            cv2.imwrite(file_name, frame)
        self.i += 1
        cv2.imshow("My webcam", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()

    minimal_image_subscriber = MinimalImageSubscriber()

    rclpy.spin(minimal_image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
