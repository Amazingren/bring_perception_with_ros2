from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='py_srv',
            executable='service',
            name='summation'
        ),
        Node(
            package='py_cli',
            executable='client',
            name='request',
            arguments = ['5', '3']
        )
    ])
