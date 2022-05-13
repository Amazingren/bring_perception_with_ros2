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
            executable='image_listener_processor',
            name='image_listener_processor',
            arguments = ['/home/bingo/Desktop/tutorial_ws/saved_images'],
            output={'stdout': 'screen', 'stderr': 'screen',}
        )
    ])
