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
            package='py_pub',
            executable='tf2_static_broadcaster',
            name='world2cam_tf_talker'
        ),
        Node(
            package='py_sub',
            executable='image_aruco_tf2_broadcaster',
            name='aruco_tf2_broadcaster',
            output='screen',
            arguments = ['--calibration', '/home/yiming/tutorial_ws/calibrationdata/ost.yaml', '--type', 'DICT_4X4_50']
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='my_visualiser'
        ),
        
    ])
