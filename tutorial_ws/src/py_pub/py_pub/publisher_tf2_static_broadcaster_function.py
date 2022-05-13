from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf_transformations as tf


class StaticFramePublisher(Node):
   """
   Broadcast static transforms.
   """
   def __init__(self):
      super().__init__('static_tf2_broadcaster')
      self._tf_publisher = StaticTransformBroadcaster(self)
      tf_msg = self.prepare_transform_msg()
      self._tf_publisher.sendTransform(tf_msg)

   def prepare_transform_msg(self):
      self.get_logger().info('I prepared a static transform message to publish to tf')
      tf_msg = TransformStamped()
      tf_msg.header.stamp = self.get_clock().now().to_msg()
      tf_msg.header.frame_id = 'world'
      tf_msg.child_frame_id = 'web_cam'

      # all the data field should be of datatype float
      tf_msg.transform.translation.x = 0.
      tf_msg.transform.translation.y = 0.
      tf_msg.transform.translation.z = 1.
      # initialise the quaternion from euler angles
      q = tf.quaternion_from_euler(0., 0., 0.)
      tf_msg.transform.rotation.x = q[0]
      tf_msg.transform.rotation.y = q[1]
      tf_msg.transform.rotation.z = q[2]
      tf_msg.transform.rotation.w = q[3]
      return tf_msg

def main():
   # pass parameters and initialize node
   rclpy.init()
   node = StaticFramePublisher()
   try:
      rclpy.spin(node)
   except KeyboardInterrupt:
      pass
   rclpy.shutdown()

if __name__ == '__main__':
    main()