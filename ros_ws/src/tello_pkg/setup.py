import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tello_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'observation_handler=tello_pkg.observation_handler:main',
            'policy_handler=tello_pkg.policy_handler:main',
            'target_handler=tello_pkg.target_handler:main',
            'vel_command_handler=tello_pkg.vel_command_handler:main',
            'mission_console=tello_pkg.mission_console:main'
        ],
    },
)
