from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='py_srv',
            executable='service',
            name='summation'
        ),
        ExecuteProcess(
        cmd=[[
        'ros2 service call ',
        '/add_two_ints ',
        'example_interfaces/srv/AddTwoInts ',
        '"{a: 3, b: 5}"'        
        ]],
        shell=True  
        )
    ])
