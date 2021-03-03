import os

from glob import glob
from setuptools import setup, find_packages

package_name = 'reachy_kinematics'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name), glob('launch/*.py')),

        (os.path.join('share', package_name), [
            'reachy.URDF', 'reachy.URDF.xacro',
        ]),

        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.dae')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Pollen Robotics',
    maintainer_email='contact@pollen-robotics.com',
    description='ROS2 packages for Reachy kinematics.',
    license='Apache-2.0 License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_kinematics_service = reachy_kinematics.arm_kinematics_service:main'
        ],
    },
)
