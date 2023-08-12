'''
Adapted from https://github.com/striest/rosbag_to_dataset/blob/tartandrive_processing/src/rosbag_to_dataset/config_parser/config_parser.py
'''
import yaml
import gymnasium as gym
from gym.spaces import Dict, Box, Discrete
import numpy as np

from collections import OrderedDict

from robot_dataset.dtypes.bool import BoolConvert
from robot_dataset.dtypes.float64 import Float64Convert
from robot_dataset.dtypes.odometry import OdometryConvert
from robot_dataset.dtypes.image import ImageConvert
from robot_dataset.dtypes.compressed_image import CompressedImageConvert
from robot_dataset.dtypes.ackermann_drive import AckermannDriveConvert
from robot_dataset.dtypes.twist import TwistConvert
from robot_dataset.dtypes.imu import ImuConvert
from robot_dataset.dtypes.pose import PoseConvert
from robot_dataset.dtypes.vector3 import Vector3Convert
from robot_dataset.dtypes.float32 import Float32Convert
# from rosbag_to_dataset.dtypes.pointcloud import PointCloudConvert
from robot_dataset.dtypes.float_stamped import FloatStampedConvert
# from rosbag_to_dataset.dtypes.racepak_sensors import RPControlsConvert, RPWheelEncodersConvert, RPShockSensorsConvert

class ConfigParser:
    """
    Class that reads in the spec of the rosbag ot convert to data.
    Expects input as a yaml file that generally looks like the following (currently WIP, subject to change).

    data:
        topic:
            type:<one of the supported types>
            folder:<output folder for this modality>
            N_per_step:<frequency factor based on dt>
            <option>:<value>
            ...
    dt: 0.1
    main_topic:<this frame is used to align the timestamp>
    """
    def __init__(self):
        pass

    def parse_from_fp(self, fp):
        x = yaml.safe_load(open(fp, 'r'))
        return self.parse(x)

    # def parse(self, spec):
    #     obs_converters = OrderedDict()
    #     outfolder = {}
    #     rates = {}
    #     import pdb;pdb.set_trace()
    #     for k,v in spec['data'].items():
    #         dtype = self.dtype_convert[spec['data'][k]['type']]
    #         converter = dtype(**spec['data'][k].get('options', {}))
    #         outfolder_k = v['folder'] if 'folder' in v.keys() else k
    #         obs_converters[k] = converter
    #         outfolder[k] = outfolder_k
    #         if 'N_per_step' in v.keys():
    #             N = spec['data'][k]['N_per_step']
    #             rates[k] = spec['dt'] / N
    #         else:
    #             rates[k] = spec['dt']
    #     if 'main_topic' in spec:
    #         maintopic = spec['main_topic']
    #     else:
    #         maintopic = list(spec['data'].keys())[0] # use first topic in the yaml file
    #     return obs_converters, outfolder, rates, spec['dt'], maintopic

    def parse(self, spec):
        obs_converters = OrderedDict()
        action_converters = OrderedDict()
        output_folders = {}
        rates = {}
        obs_spaces = {}
        action_space = None

        for key, val in spec['data'].items():
            dtype_fn = self.dtype_convert[val['type']]
            converter = dtype_fn(**val.get('options', {}))
            out_folder = val['folder'] if 'folder' in val.keys() else key
            output_folders[key] = out_folder

            if val.get('options', {}).get('mode') == "action":
                action_converters[key] = converter
                # action_space[out_folder] = converter.action_space()
                action_space = converter.action_space()
            else:
                obs_converters[key] = converter
                obs_space = converter.obs_space()
                obs_spaces[out_folder] = obs_space

            if 'N_per_step' in val.keys():
                N = val['N_per_step']
                rates[key] = spec['dt'] / N
            else:
                rates[key] = spec['dt']

        obs_spaces = gym.spaces.Dict(obs_spaces)
        # action_space = gym.spaces.Dict(action_space)

        if 'main_topic' in spec:
            main_topic = spec['main_topic']
        else:
            main_topic = list(spec['data'].keys())[0] # use first topic in the yaml file
        return obs_converters, action_converters, output_folders, rates, spec['dt'], main_topic, obs_spaces, action_space


    dtype_convert = {
        "AckermannDrive":AckermannDriveConvert,
        "Bool":BoolConvert,
        "CompressedImage":CompressedImageConvert,
        "Float64":Float64Convert,
        "Image":ImageConvert,
        "Imu":ImuConvert,
        "Odometry":OdometryConvert,
        # "PointCloud":PointCloudConvert,
        "Pose":PoseConvert,
        "Twist":TwistConvert,
        "Vector3":Vector3Convert,
        "FloatStamped":FloatStampedConvert,
        "Float32":Float32Convert,
        # "RPControls":RPControlsConvert,
        # "RPWheelEncoders":RPWheelEncodersConvert,
        # "RPShockSensors":RPShockSensorsConvert,
    }

# class ParseObject:
#     """
#     Basically a dummy class that has an observation_space and action_space field.
#     """
#     def __init__(self, observation_space, action_space, dt):
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.dt = dt

if __name__ == "__main__":
    fp = open('/home/mateo/robot_dataset/specs/sample_tartandrive.yaml')
    d = yaml.safe_load(fp)
    print("=====")
    print("YAML Spec: ")
    print(d)
    print("-----")
    print(f"Type of spec: {type(d)}")
    parser = ConfigParser()
    obs_converters, output_folders, rates, dt, main_topic, obs_spaces, action_space = parser.parse(d)
    print("=====")
    print("Observation converter (OrderedDict): ")
    print(obs_converters)
    print("-----")
    print("Output folders: ")
    print(output_folders)
    print("-----")
    print("Rates: ")
    print(rates)
    print("-----")
    print("Main timestep: ")
    print(dt)
    print("-----")
    print("Main topic: ")
    print(main_topic)
    print("-----")
    print("obs_spaces")
    print(obs_spaces)
    print("-----")
    print("action_space")
    print(action_space)
