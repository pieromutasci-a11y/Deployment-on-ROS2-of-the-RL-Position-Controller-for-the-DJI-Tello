import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tello_pkg_web'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']) + [
        package_name + '.web_static',
        package_name + '.web_static.vendor',
    ],
    package_data={
        package_name: [
            'web_static/*',
            'web_static/vendor/*',
        ],
    },
    include_package_data=True,
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
    description='Controllore di posizione Tello con interfaccia web (dashboard 3D + WebSocket).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vel_command_handler_web=tello_pkg_web.vel_command_handler_web:main',
        ],
    },
)
