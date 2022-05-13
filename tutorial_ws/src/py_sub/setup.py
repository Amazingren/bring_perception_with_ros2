from setuptools import setup

package_name = 'py_sub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yiming',
    maintainer_email='ywang@fbk.eu',
    description='Example of the minimal subscriber using rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'listener = py_sub.subscriber_member_function:main',
        'image_listener = py_sub.image_subscriber_function:main',
        'student_listener = py_sub.student_subscriber_function:main',
        'image_listener_processor = py_sub.image_subscriber_processing_function:main',
        'image_aruco_estimator = py_sub.image_subscriber_aruco_function:main',
        'image_aruco_tf2_broadcaster = py_sub.image_subscriber_aruco_tf2_broadcast_function:main'
        ],
    },
)
