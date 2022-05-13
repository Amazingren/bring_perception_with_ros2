from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_pub',
            executable='image_talker',
            name='image_talker'
        ),
        Node(
            package='py_sub',
            executable='image_listener',
            name='image_listener',
            arguments = ['/home/bingo/tutorial_ws/saved_images']
        )
    ])
