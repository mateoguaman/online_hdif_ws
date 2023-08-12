from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['physics_atv_deep_stereo_vo', 'physics_atv_deep_stereo_vo.PSM', 'physics_atv_deep_stereo_vo.PWC'],
  package_dir={'': 'src'}
)

setup(**d)
