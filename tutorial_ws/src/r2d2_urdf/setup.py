import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'r2d2_urdf'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('urdf/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer_email='ywang@fbk.eu',
    description='Example of the robot state publisher using rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'r2d2_state_publisher = r2d2_urdf.r2d2_state_publisher:main'
        ],
    },
)
