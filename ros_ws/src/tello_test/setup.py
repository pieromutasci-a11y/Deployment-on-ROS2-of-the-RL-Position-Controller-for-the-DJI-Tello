from setuptools import find_packages, setup

package_name = 'tello_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Nodi di test hardware per il drone Tello (takeoff/land/validazione assi, lettura sensori).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'read_sensors=tello_test.read_sensors:main',
            'takeoff_land=tello_test.takeoff_land:main',
        ],
    },
)
