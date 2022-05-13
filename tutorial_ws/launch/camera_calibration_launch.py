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
            package='camera_calibration',
            executable='cameracalibrator',
            name='calibrator',
            remappings= [('/image', '/web_cam')],
            arguments = ['--size', '8x6', '--square', '0.023']
        )
    ])
