# physics_atv_local_mapping

## Compile the code in the Release mode
```
catkin build -DCMAKE_BUILD_TYPE=Release
```

## Run the nodes
- Update the physics_atv_deep_stereo_vo package to the latest version, and use the `python3` branch. 
```
cd physics_atv_deep_stereo_vo
git pull
git checkout python3
```

- Launch the nodes
```
roslaunch physics_atv_deep_stereo_vo multisense_register_localmapping.launch
```

- Play a bagfile (Note: if you see warning message of VO/Stereo nodes are skipping frames, you'd better decrease the data frequency for best performance. )

## Visualize the map
```
cd physics_atv_local_mapping/src
python test_show_localmaps.py
```

## Parameters
These parameters can be found in physics_atv_deep_stereo_vo/launch/multisense_register_localmapping.launch

- Mapping resolution: [physics_atv_local_mapping] `/resolution`

- Mapping range: [physics_atv_deep_stereo_vo] `/pc_max_dist` and `/pc_min_dist`, [physics_atv_local_mapping] `max_x`, `min_x`, `max_y`, `min_y`. (Note: `pc_max_dist` should be larger than `max_x`) 

- Color image source: [physics_atv_deep_stereo_vo] `/image_rect` should be set to true if `/multisense/left/image_rect_color` is available, or set to false if we only have `/multisense/left/image_color`. 

## Topics
The output of the mapping nodes are top-down RGB map and height map. An example can be found in test_show_localmaps.py about converting the ROS message to the numpy array. 

- `/local_height_map`: Top-down height map, size `H x W x 2` (min/max), type `Image`, encoding `32FC2`. 

- `/local_height_map_inflate`: Top-down height map with simple hole filling, size `H x W x 2` (min/max), type `Image`, encoding `32FC2`. 

- `/local_rgb_map`: Top-down RGB map, size `H x W x 3` (RGB), type `Image`, encoding `rgb8`. 

- `/local_rgb_map_inflate`: Top-down RGB map with simple hole filling, size `H x W x 3` (RGB), type `Image`, encoding `rgb8`. 


## For the python version (deprecated)

### Compile the Cython code
```
cd src
python setup.py build_ext --inplace
```

### Run the node
- Launch the physics_atv_deep_stereo_vo node, which generates colored point cloud from stereo image
```
roslaunch physics_atv_deep_stereo_vo multisense_localmapping.launch
```

- Run the localmapping node
```
cd src
python LocalMapping.py
```

### Run local mapping with multisense point cloud
```
roslaunch physics_atv_local_mapping multisense_sgm.launch
```

The topdown map is published as `/local_height_map` and `/local_rgb_map`. 