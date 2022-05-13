from setuptools import setup

package_name = 'py_srv'

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
    description='Python server example',
    license='Apache Liscense 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'service = py_srv.service_member_function:main',
        ],
    },
)
