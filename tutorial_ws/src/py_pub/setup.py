from setuptools import setup

package_name = 'py_pub'

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
    description='Example of the minimal publisher using rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'talker = py_pub.publisher_member_function:main',
        'student_talker = py_pub.student_publisher_function:main',
        'image_talker = py_pub.image_publisher_function:main',
        'tf2_static_broadcaster = py_pub.publisher_tf2_static_broadcaster_function:main',
        ],
    },
)
