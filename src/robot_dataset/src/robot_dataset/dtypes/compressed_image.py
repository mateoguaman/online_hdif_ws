import rospy
import numpy as np
import cv2

import gymnasium as gym
from gymnasium.spaces import Space, Dict, Discrete, Box

from sensor_msgs.msg import CompressedImage

from rosbag_to_dataset.dtypes.base import Dtype

class CompressedImageConvert(Dtype):
    """
    For image, we'll rescale and 
    """
    def __init__(self, nchannels, output_resolution, is_compressed=False):
        """
        Args:
            nchannels: The number of channels in the image
            output_resolution: The size to rescale the image to
            aggregate: One of {'none', 'bigendian', 'littleendian'}. Whether to leave the number of channels alone, or to combine with MSB left-to-right or right-to-left respectively.
            empty_value: A value signifying no data. Replace with the 99th percentile value to make learning simpler.
        """
        self.nchannels = nchannels
        self.output_resolution = output_resolution

    def N(self):
        return self.output_resolution + [self.nchannels] 
    
    def obs_space(self):
        dims = self.output_resolution + [self.nchannels]
        return Box(low=0, high=255, shape=dims, dtype=np.uint8)
    
    def action_space(self):
        return None

    def rosmsg_type(self):
        return CompressedImage

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

        data = np.frombuffer(msg.data, dtype=np.uint8).copy()

        data = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

        if self.output_resolution[0] == 0 or self.output_resolution[1] == 0:
            return data

        new_height = self.output_resolution[0]
        new_width = self.output_resolution[1]
        data = cv2.resize(data, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)
        # Make sure 1-channel images preserve channel dimension
        data = data.reshape(new_height, new_width, self.nchannels)

        # data = np.moveaxis(data, 2, 0) #Switch to channels-first

        # data = data.astype(np.float32) / (255.)

        return data
    

    def save_file_one_msg(self, msg, filename):
        """
        Save the data to hard drive.
        This function should be implemented where the data is stored frame by frame like image or point cloud
        """
        data = self.ros_to_numpy(msg)
        cv2.imwrite(filename+'.png', data)


if __name__ == "__main__":
    c = CompressedImageConvert(nchannels=1, output_resolution=[32, 32])
    # msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    # print(c.ros_to_numpy(msg))
