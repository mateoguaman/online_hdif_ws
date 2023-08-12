#!/usr/bin/python3
import rospy
import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import wandb
import pickle
from functools import partial
import yaml
import time

from robot_dataset.data.replay_buffer import ReplayBuffer
from robot_dataset.online_converter.robot_listener import RobotListener
from robot_dataset.utils.utils import quaternionToEuler
from robot_dataset.models.mlp import MLP
from robot_dataset.models.resnet import ResNet
# from robot_dataset.models.hdif import CostFourierVelModel
from learned_cost_map.trainer.model import CostFourierVelModel
from learned_cost_map.utils.costmap_utils import rosmsgs_to_maps_batch, produce_training_input
from robot_dataset.agents.mse import MSE

USE_WANDB = True


def transform(batch, map_metadata, crop_params, fourier_freqs=None, vel=None):
    # import pdb;pdb.set_trace()
    obs = batch['observations']
    next_obs = batch['next_observations']
    actions = torch.from_numpy(batch['actions'])

    odom = obs['state']
    rgb_map = obs['rgb_map']
    height_map = obs['height_map']
    traversability_cost = obs["traversability_cost"]

    if vel == None:
        vel = np.linalg.norm(odom[:, 7:10], axis=1)
    else:
        vel = 3  ## Set to 3 m/s

    maps = rosmsgs_to_maps_batch(rgb_map, height_map)
    input_data = produce_training_input(maps, map_metadata, crop_params, vel=vel, fourier_freqs=fourier_freqs)

    ground_truth = torch.from_numpy(traversability_cost)

    return input_data, ground_truth



def main():
    ## Define data source (RobotListener or Simulation Environment)
    # Load spec and get parser
    config_spec = "/home/mateo/local_phoenix_ws/src/robot_dataset/specs/hdif_lester.yaml"
    map_config = "/home/mateo/local_phoenix_ws/src/learned_cost_map/configs/map_params.yaml"
    saved_model = "/home/mateo/local_phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/epoch_50.pt"
    saved_freqs = "/home/mateo/local_phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/fourier_freqs.pt"
    # import pdb;pdb.set_trace()
    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    rospy.init_node('online_hdif')
    dt = 0.1
    rate = rospy.Rate(int(1/dt))
    robot_listener = RobotListener(config_spec=config_spec)
    print("Robot Listener set up")

    ## Define agent similar to BC (take in model, take in batch, return MSE loss)
    model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, pretrained=False)
    model.load_state_dict(torch.load(saved_model))
    fourier_freqs = torch.load(saved_freqs)
    print("Model defined")

    ## Define data transform
    data_transform = partial(transform, map_metadata=map_metadata, crop_params=crop_params, fourier_freqs=fourier_freqs)
    
    ## Define agent
    agent = MSE() 
    print("Agent defined")

    model.to("cuda")
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    ## Instantiate replay buffer 
    buffer_capacity = 200
    replay_buffer = ReplayBuffer(robot_listener.obs_space, robot_listener.action_space, buffer_capacity)
    print("Replay Buffer defined")

    ## Training loop. Is there a way to make it asynchronous? Threading? One for gathering data into replay buffer, one for training
    lifetime = 1000000
    count = 1
    replay_ratio = 10  ## For now, it is the number of batches obtained from the dataloader and trained at every training step within data collection
    batch_size = 64
    training_freq = 10
    save_freq = 10
    save_dir = "/media/mateo/MateoSSD/online_hdif"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config = {
        'lr': lr,
        'buffer_capacity': buffer_capacity, 
        'replay_ratio': replay_ratio,
        'batch_size': batch_size,
        'training_freq': training_freq,
        'save_freq': save_freq
    }
    wandb.init(project="online_learning", reinit=True, config=config)

    print('waiting 2s for topics...')
    for i in range(10):
        rate.sleep()

    # with open("/media/mateo/MateoSSD/online_hdif/datasets/buffer_200.pickle", 'rb') as pickle_file:
    #     replay_buffer = pickle.load(pickle_file)
    # replay_buffer = pickle.load("/media/mateo/MateoSSD/online_hdif/datasets/buffer_200.pickle")
    # batch = replay_buffer.sample(batch_size=batch_size)
    # train_metrics = agent.update(model, batch, data_transform, optimizer)
    # loss = train_metrics["loss"]
    # print(f"Training loss: {loss}")

    while (not rospy.is_shutdown()) and (count < lifetime):
        print(f"Iteration {count}/{lifetime}")
        robot_data = robot_listener.get_data()
        replay_buffer.insert(robot_data)

        if count % training_freq == 0:  ## Train network
            ## Extract batches in for loop
            for i in range(replay_ratio):
                batch = replay_buffer.sample(batch_size=batch_size)
                train_metrics = agent.update(model, batch, data_transform, optimizer)
                loss = train_metrics["loss"]
                print(f"Training loss: {loss}")
                wandb.log(data=train_metrics, step=count)

        ## Save model and buffer every k iterations
        if (count % save_freq == 0):
            buffer_folder = os.path.join(save_dir, "datasets")
            if not os.path.exists(buffer_folder):
                os.makedirs(buffer_folder)
            dataset_file = os.path.join(buffer_folder, f"buffer_{count}.pickle")
            model_folder = os.path.join(save_dir, "models")
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_file = os.path.join(model_folder, f"model_{count}.pt")
            with open(dataset_file, "wb") as f:
                pickle.dump(replay_buffer, f)

            torch.save(model.state_dict(), model_file)
        count += 1
        # rate.sleep()
        time.sleep(dt)

if __name__ == "__main__":
    main()