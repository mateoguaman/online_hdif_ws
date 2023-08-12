#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import jax 
import jax.numpy as jnp
import time
import gymnasium as gym
from gymnasium.spaces import Space, Dict, Discrete, Box

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from cv_bridge import CvBridge
from celluloid import Camera
from robot_dataset.data.replay_buffer import ReplayBuffer
from robot_dataset.config_parser.config_parser import ConfigParser


class RobotListener(object):
    def __init__(self, config_spec, buffer_capacity=10000, use_stamps=True):
        # '''
        # Args:
        #     - topics: List of Topic objects
        # '''
        # self.topics = topics
        # self.buffer_capacity = buffer_capacity
        # self.cvbridge = CvBridge()
        # self.subscribers = [
        #     rospy.Subscriber(self.topics[i].topic_name, 
        #                      self.topics[i].msg_type, 
        #                      self.topics[i].io_function, 
        #                      queue_size=1) for i in range(len(self.topics)
        #                     )]
        
        # self.obs_space, self.act_space = _obs_act_space_from_topics(self.topics)
        
        # self.replay_buffer = ReplayBuffer(self.obs_space, self.act_space, buffer_capacity)

        self.config_spec = config_spec
        self.buffer_capacity = buffer_capacity
        self.use_stamps = use_stamps
        self.max_queue_len = 100

        parser = ConfigParser()
        self.obs_converters, self.action_converters, self.output_names, self.rates, self.dt, self.main_topic, self.obs_space, self.action_space = parser.parse_from_fp(self.config_spec)
        # TODO: Should make max_queue_len equal to max_rate/min_rate for efficiency

        # self.replay_buffer = ReplayBuffer(self.obs_space, self.action_space, self.buffer_capacity)
        self.init_subscribers()
        self.init_queue()

        
    def init_subscribers(self):
        self.subscribers = {}
        for k in self.obs_converters.keys():
            self.subscribers[k] = rospy.Subscriber(k, self.obs_converters[k].rosmsg_type(), self.handle_msg, callback_args=k)

        for k in self.action_converters.keys():
            self.subscribers[k] = rospy.Subscriber(k, self.action_converters[k].rosmsg_type(), self.handle_msg, callback_args=k)

    # def init_queue(self):
    #     self.queue = {}  
    #     self.times = {} 
    #     self.curr_time = rospy.Time.now()
    #     for topic, converter in self.obs_converters.items():
    #         if self.rates[topic] == self.dt:
    #             self.queue[topic] = None
    #         else:
    #             self.queue[topic] = [None] * self.max_queue_len
    #             self.times[topic] = [0.] * self.max_queue_len

    #     for topic, converter in self.action_converters.items():
    #         self.queue[topic] = None

    def init_queue(self):
        self.queue = {}  
        self.times = {} 
        self.curr_time = rospy.Time.now()
        for topic, converter in self.obs_converters.items():
            if self.rates[topic] == self.dt:
                self.queue[topic] = [None] * 2  # Will keep two timesteps, for current and next observations.
            else:
                self.queue[topic] = [None] * 2 * self.max_queue_len
                self.times[topic] = [0.] * 2 * self.max_queue_len

        for topic, converter in self.action_converters.items():
            self.queue[topic] = [None]

    # def handle_msg(self, msg, topic):
    #     has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
    #     has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

    #     if self.use_stamps and (has_stamp or has_info):
    #         t = msg.header.stamp if has_stamp else msg.info.header.stamp
    #     else:
    #         t = rospy.Time.now()

    #     if self.rates[topic] == self.dt:
    #         self.queue[topic] = msg
    #     else:
    #         self.queue[topic] = self.queue[topic][1:] + [msg]
    #         self.times[topic] = self.times[topic][1:] + [t]

    #     # If main topic, add to replay buffer

    def handle_msg(self, msg, topic):
        has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
        has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

        if self.use_stamps and (has_stamp or has_info):
            t = msg.header.stamp if has_stamp else msg.info.header.stamp
        else:
            t = rospy.Time.now()

        if self.rates[topic] == self.dt:
            self.queue[topic] = self.queue[topic][1:] + [msg]
        else:
            self.queue[topic] = self.queue[topic][1:] + [msg]
            self.times[topic] = self.times[topic][1:] + [t]

        # If main topic, add to replay buffer

    # def get_data(self):
    #     out = {
    #         'observations': {},
    #         'actions': None,
    #         'next_observations': {},
    #         'rewards': None,
    #         'masks': None,
    #         'dones': None
    #     }

    #     for topic, converter in self.obs_converters.items():
    #         if self.rates[topic] == self.dt:
    #             out['observations'][self.output_names[topic]] = converter.ros_to_numpy(self.queue[topic])
    #         else:
    #             # Get the times in seconds at which we have messages in the queue for a given topic
    #             msg_times = np.array([t.to_sec() for t in self.times[topic]])
    #             # Look at when the last message was received, and get a target range of times [last-dt, ..., last] that would ideally correspond to the times of a sequence of data
    #             target_times = np.arange(msg_times[-1] - self.dt, msg_times[-1], self.rates[topic])
    #             # Creates a matrix of element-wise distances from all members of target_times wo all members of msg_times
    #             dists = abs(np.expand_dims(target_times, 0) - np.expand_dims(msg_times, 1))
    #             msg_idxs = np.argmin(dists, axis=0)
    #             datas = [converter.ros_to_numpy(self.queue[topic[i]]) for i in msg_idxs]
    #             out['observations'][self.output_names[topic]] = np.stack([x for x in datas], dim=0)

    #     for topic, converter in self.action_converters.items():
    #         # out['actions'][self.output_names[topic]] = converter.ros_to_numpy(self.queue[topic])
    #         out['actions'] = converter.ros_to_numpy(self.queue[topic])

    #     out['next_observations'] = out['observations']

    #     return out

    def get_data(self):
        out = {
            'observations': {},
            'actions': None,
            'next_observations': {},
            'rewards': None,
            'masks': None,
            'dones': None
        }

        for topic, converter in self.obs_converters.items():
            if self.rates[topic] == self.dt:
                out['observations'][self.output_names[topic]] = converter.ros_to_numpy(self.queue[topic][0])
                out['next_observations'][self.output_names[topic]] = converter.ros_to_numpy(self.queue[topic][1])
            else:
                num_msgs = round(1/self.rates[topic])
                # Get the times in seconds at which we have messages in the queue for a given topic
                msg_times = np.array([t.to_sec() for t in self.times[topic]])
                # Look at when the last message was received, and get a target range of times [last-dt, ..., last] that would ideally correspond to the times of a sequence of data
                target_times = np.arange(msg_times[-1] - self.dt*2, msg_times[-1], self.rates[topic])
                # Creates a matrix of element-wise distances from all members of target_times wo all members of msg_times
                dists = abs(np.expand_dims(target_times, 0) - np.expand_dims(msg_times, 1))
                msg_idxs = np.argmin(dists, axis=0)
                datas = [converter.ros_to_numpy(self.queue[topic[i]]) for i in msg_idxs]
                out['observations'][self.output_names[topic]] = np.stack([x for x in datas[:num_msgs]], dim=0)
                out['next_observations'][self.output_names[topic]] = np.stack([x for x in datas[-num_msgs:]], dim=0)

        for topic, converter in self.action_converters.items():
            # out['actions'][self.output_names[topic]] = converter.ros_to_numpy(self.queue[topic])
            out['actions'] = converter.ros_to_numpy(self.queue[topic][0])

        # out['next_observations'] = out['observations']

        return out

if __name__ == "__main__":
    # Load spec and get parser
    config_spec = "/home/mateo/local_phoenix_ws/src/robot_dataset/specs/hdif_lester.yaml"

    rospy.init_node('robot_listener')
    rate = rospy.Rate(10)

    robot_listener = RobotListener(config_spec=config_spec)

    print("Observation space: ")
    print(robot_listener.obs_space)
    print("Action space: ")
    print(robot_listener.action_space)

    buffer_capacity = 100
    replay_buffer = ReplayBuffer(robot_listener.obs_space, robot_listener.action_space, buffer_capacity)

    # TODO: This should be made more robust: it should either not start until all topics are present or the robot listener should only try to process messages when they are not None
    print('waiting 2s for topics...')
    for i in range(10):
        rate.sleep()

    while (not rospy.is_shutdown()) and len(replay_buffer)<buffer_capacity:
        robot_data = robot_listener.get_data()
        # print(robot_data)
        replay_buffer.insert(robot_data)
        print(f"robot_buffer has size: {len(replay_buffer)}")

        rate.sleep()

    import pdb;pdb.set_trace()

    images = replay_buffer.dataset_dict['observations']['image_left_color']
    actions = replay_buffer.dataset_dict['actions']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    camera = Camera(fig)
    # import matplotlib.pyplot as plt
    for i, image in enumerate(images):
        print(f"Step {i}")
        # ax.set_title(f"Step {i}")
        ax.text(0.5, 1.01, f'Step {i}', transform=ax.transAxes)
        plt.imshow(image)
        # plt.close()
        camera.snap()
        # plt.pause(0.1)
    animation = camera.animate()
    # plt.show()
    animation.save("replay_buffer.mp4")
    plt.close()
    plt.plot(range(buffer_capacity), actions[:,0], label="throttle")
    plt.plot(range(buffer_capacity), actions[:, 1], label="steer")
    plt.legend()
    plt.grid()
    plt.title("Actions in replay buffer")
    plt.savefig("action_replay.png", dpi=300, bbox_inches="tight")
    plt.show()