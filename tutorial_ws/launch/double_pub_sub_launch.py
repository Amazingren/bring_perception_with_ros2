from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_pub',
            namespace = 'pubsub1',
            executable='talker',
            name='talker'
        ),
        Node(
            package='py_sub',
            namespace = 'pubsub1',
            executable='listener',
            name='listener'
        ),
        Node(
            package='py_pub',
            namespace = 'pubsub2',
            executable='talker',
            name='talker'
        ),
        Node(
            package='py_sub',
            namespace = 'pubsub2',
            executable='listener',
            name='listener'
        )
    ])
