from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_pub',
            executable='talker',
            name='talker'
        ),
        Node(
            package='py_sub',
            executable='listener',
            name='listener'
        )
    ])
