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
            executable='image_aruco_estimator',
            name='image_aruco_estimator',
            output='screen',
            arguments = ['--calibration', '/home/yiming/tutorial_ws/calibrationdata/ost.yaml', '--type', 'DICT_4X4_50']
        )
    ])
