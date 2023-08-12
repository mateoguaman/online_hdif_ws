from distutils.core import setup
## Uncomment if using catkin build (not just python)
from catkin_pkg.python_setup import generate_distutils_setup
import numpy

## Uncomment if using pip install 
# setup(name='robot_dataset',
#       version='1.0',
#       description='',
#       author='',
#       author_email='',
#       url='',
#       include_dirs=[numpy.get_include()],
#       packages=['robot_dataset', 
#             'robot_dataset.agents',
#             'robot_dataset.config_parser', 
#             'robot_dataset.data',  
#             'robot_dataset.dtypes',
#             'robot_dataset.models',
#             'robot_dataset.online_converter',
#             'robot_dataset.utils'
#             ],
#       package_dir={'': 'src'},
#       # zip_safe=False
#      )

# Uncomment if using catkin build
d = generate_distutils_setup(
    packages=['robot_dataset', 
            'robot_dataset.agents',
            'robot_dataset.config_parser', 
            'robot_dataset.data',  
            'robot_dataset.dtypes',
            'robot_dataset.models',
            'robot_dataset.online_converter',
            'robot_dataset.utils'
            ],
      package_dir={'': 'src'},
)

setup(**d)