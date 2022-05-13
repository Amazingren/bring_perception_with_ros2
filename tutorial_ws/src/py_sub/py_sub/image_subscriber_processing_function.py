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
import tensorflow as tf
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from mtcnn.mtcnn import MTCNN
import copy
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
        self.detector = MTCNN()
        # If a data path is provided, save the frames
        if len(sys.argv)>=2:
            self.get_logger().info('There are %d args' % len(sys.argv) )
            self.save_path = sys.argv[1]
            # check if the folder exists. If not, create one
            if not os.path.exists(self.save_path):
                self.get_logger().info('Create the folder for saving the video frames')
                os.makedirs(self.save_path)

        #### For Face ID
        self.model = tf.keras.models.load_model("/home/bingo/Desktop/tutorial_ws/src/py_sub/cosface.h5")
        self.gallery_features = np.load('/home/bingo/Desktop/tutorial_ws/src/py_sub/py_sub/features.npy')
        self.gallery_file_list = np.load('/home/bingo/Desktop/tutorial_ws/src/py_sub/py_sub/file_names.npy')

    def decode_img(self, img, img_height=112, img_width=112):
        shapes = np.array(img.shape, dtype=np.float)[:-1]
        big_side = max(shapes)
        new_side = np.ceil(big_side / 2) * 2
        diff = new_side - shapes
        half_diff = diff // 2
        top, left = diff - half_diff
        bottom, right = half_diff
        img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT)
        return tf.image.resize(img, [img_height, img_width])

    def normalize_image(self, img):
        return (img - 127.5) / 128

    @tf.function
    def extract_features(self, images):
        features = self.model(images, training=False)
        return features

    def extract_feature_image(self, image):
        tmp_img = self.normalize_image(self.decode_img(image))
        return self.extract_features(tf.expand_dims(tmp_img,axis=0))

    def listener_callback(self, msg):
        self.get_logger().info('I received %d Image messages with frame id %s with width %d and height %d.'
                               % (self.i, msg.header.frame_id, msg.width, msg.height))
        sender_timestamp_msg = msg.header.stamp
        print('The message is received at: %s' % datetime.utcfromtimestamp(sender_timestamp_msg.sec).isoformat())

        # extract the image frame from the Image msg
        frame = self.cvbr.imgmsg_to_cv2(msg)
        result_frame = copy.deepcopy(frame)
        # perform face detection
        faces = self.detector.detect_faces(frame)
        for face_count, face in enumerate(faces):
            bounding_box = face['box']
            cv2.rectangle(result_frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2)
            face_image = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

            # Face ID
            max = 0
            min = 2000000
            if len(face_image) > 0:
                query_sample = self.extract_feature_image(face_image)
                print("######################")
                print(query_sample.shape)
                print(face_image.shape)
                
                query_sample_tiles = tf.tile(query_sample, [len(self.gallery_features), 1])
                dists = tf.math.sqrt(tf.reduce_sum(tf.math.square(query_sample_tiles - self.gallery_features), axis=-1))
                if tf.reduce_max(dists) > tf.cast(max, tf.float32):
                    max = tf.reduce_max(dists).numpy()
                if tf.reduce_min(dists) < tf.cast(min, tf.float32):
                    min = tf.reduce_min(dists).numpy()
                rank = tf.argsort(
                    dists, axis=-1, direction='ASCENDING', stable=False, name=None
                )
                gallery_list = []
                for j, r in enumerate(rank):
                    gallery_list.append(os.path.basename(self.gallery_file_list[r]))
                results = gallery_list[0]
                cv2.putText(result_frame, results, (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 5)
                print(results)
        self.i += 1


        # Visuilization
        cv2.imshow("My webcam", result_frame)
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
