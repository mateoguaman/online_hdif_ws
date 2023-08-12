# physics_atv_deep_stereo_vo
Use deep learning model for stereo matching and visual odometry

## Setup
Download the models: 
```
sh download_models.sh
```

Python dependencies: 
- Python 2
- numpy
- pytorch >= 1.3
- opencv-python
- cupy

## Run with a bagfile
```
rosparam set use_sim_time true
roslaunch physics_atv_deep_stereo_vo multisense_vo.launch 
rosbag play FILENAME.bag -r 0.3 --clock
```

Open the rviz with the config file stereovo.rviz in the root folder.

## TODO
- Project colored image onto pointcloud
- Combine Stereo and VO
- Optimize, Speed up