from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_pub',
            executable='student_talker',
            name='talker'
        ),
        Node(
            package='py_sub',
            executable='student_listener',
            name='listener'
        )
    ])
