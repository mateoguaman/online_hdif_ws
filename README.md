Using this catkin workspace, I run it as follows:
0. Build this catkin workspace with ```catkin build``` and source with ```source devel/setup.bash```. Source in all terminals.
1. Terminal 1: run ```roscore```
2. Terminal 2: run ```rosparam set use_sim_time true```
3. Terminal 2: Run a rosbag from lester in the background. I use this loop one: ```s3://arl-aimm-data/warthog/2023-05-05/gq_cmu_forest_loop_data_collect_04_2023-05-05-13-40-11.bag``` with the following command: ```rosbag play --clock gq_cmu_forest_loop_data_collect_04_2023-05-05-13-40-11.bag```
4. Terminal 3: run ```roslaunch learned_cost_map cmu_sara_stack_lester.launch```
5. Terminal 4: run ```python robot_dataset/scripts/online_hdif.py```. Make sure to set the right parameters for where the pre-trained models are stored and where the models will be stored. These should ideally be put into an argparser or a hydra config.
6. Make sure you can visualize training loss with wandb.