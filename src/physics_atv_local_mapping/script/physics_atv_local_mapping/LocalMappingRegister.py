#!/usr/bin/env python3

import numpy as np

import rospy

from sensor_msgs.point_cloud2 import PointCloud2, PointField
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import message_filters
# from cv_bridge import CvBridge

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ScrollGrid import ScrollGrid
from GridFilter import GridFilter

from scipy.spatial.transform import Rotation
import time
from utils import pointcloud2_to_xyzrgb_array, xyz_array_to_point_cloud_msg



def coord_transform(points, R, T):
    points_trans = np.matmul(R.transpose(1,0), points.transpose(1,0)) - T
    return points_trans.transpose(1, 0)

def transform_ground(points):
    # the R and T come from gound calibration
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    # starttime = time.time()
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    # rospy.loginfo("transform ground pt num {} time {}".format(points.shape[0], time.time()-starttime))
    return points_trans.transpose(1, 0)


class LocalMappingRegisterNode(object):
    def __init__(self):

        pc_sync = message_filters.Subscriber('/statistical_outlier_removal/output', PointCloud2)
        trans_sync = message_filters.Subscriber('/tartanvo_transform', TransformStamped)
        ts = message_filters.TimeSynchronizer([pc_sync, trans_sync], 100)
        ts.registerCallback(self.handle_pc)

        self.xyz_register = None
        self.color_register = None

        self.resolution = rospy.get_param('~resolution', 0.05)
        self.min_x = rospy.get_param('~min_x', 0.0)
        self.max_x = rospy.get_param('~max_x', 10.0)
        self.min_y = rospy.get_param('~min_y', -5.)
        self.max_y = rospy.get_param('~max_y', 5.)
        self.max_points_num = rospy.get_param('~max_points_num', 1000000) # throw away old frames
        self.visualize_maps = rospy.get_param('~visualize_maps', True)

        self.localmap = ScrollGrid(self.resolution, (self.min_x, self.max_x, self.min_y, self.max_y))

        self.height_pub_ = rospy.Publisher('local_height_map', GridMap, queue_size=1)
        self.rgb_pub_ = rospy.Publisher('local_rgb_map', Image, queue_size=1)
        # self.cvbridge = CvBridge()

        # self.gridfilter = GridFilter(self.resolution/2, (self.min_x, self.max_x, self.min_y, self.max_y, -0.5, 1.5))

        # for debugging
        self.cloud_pub_ = rospy.Publisher('deep_cloud_register', PointCloud2, queue_size=1)


    def points_filter(self, points_3d, points, colors):
        # import ipdb;ipdb.set_trace()
        # points = points.reshape(-1, 3)
        mask = points_3d[:,0] > self.min_x - 2
        points_filter = points[mask, :]
        colors_filter = colors[mask, :]
        points_3d_filterd = points_3d[mask, :]
        print('Points filter: {} - {}'.format(points_3d.shape[0], points_3d_filterd.shape[0]))
        return points_3d_filterd[:self.max_points_num, :], points_filter[:self.max_points_num, :], colors_filter[:self.max_points_num, :]

    def to_gridmap_rosmsg(self, data, stamp):
        '''
        data: heightmap: h x w x 2, first channel min-height, second channel max-height
        '''
        msg = GridMap()
        msg.info.header.stamp = stamp
        msg.info.header.frame_id = "multisense"
        msg.info.resolution = self.resolution
        msg.info.length_x = self.max_x - self.min_x
        msg.info.length_y = self.max_y - self.min_y
        msg.layers = ["low", "high"]

        data_msg_low = Float32MultiArray()
        data_msg_low.layout.dim = [MultiArrayDimension("column_index", data.shape[0], data.shape[0] * data.dtype.itemsize), MultiArrayDimension("row_index", data.shape[1], data.shape[1] * data.dtype.itemsize)]
        data_msg_low.data = data[:,:,0].reshape([1, -1])[0].tolist()

        data_msg_high = Float32MultiArray()
        data_msg_high.layout.dim = [MultiArrayDimension("column_index", data.shape[0], data.shape[0] * data.dtype.itemsize), MultiArrayDimension("row_index", data.shape[1], data.shape[1] * data.dtype.itemsize)]
        data_msg_high.data = data[:,:,1].reshape([1, -1])[0].tolist()

        msg.data = [data_msg_low, data_msg_high]
        return msg

    def handle_pc(self, pc_msg, trans_msg):
        # print('point cloud callback...')
        starttime = time.time()
        xyz_array, color_array = pointcloud2_to_xyzrgb_array(pc_msg)
        if self.xyz_register is not None:
            T = np.array([[trans_msg.transform.translation.x], 
                          [trans_msg.transform.translation.y],
                          [trans_msg.transform.translation.z]])
            quat = np.array([trans_msg.transform.rotation.x, 
                             trans_msg.transform.rotation.y,
                             trans_msg.transform.rotation.z,
                             trans_msg.transform.rotation.w])
            R = Rotation.from_quat(quat).as_dcm()
            points_trans = coord_transform(self.xyz_register, R, T)
            self.xyz_register = np.concatenate((xyz_array, points_trans),axis=0)
            self.color_register = np.concatenate((color_array, self.color_register),axis=0)
            xyz_register_ground = transform_ground(self.xyz_register)

            xyz_register_ground, self.xyz_register, self.color_register = self.points_filter(xyz_register_ground, self.xyz_register, self.color_register)
            # # filter by density
            # filter_mask = self.gridfilter.grid_filter(xyz_register_ground)
            # xyz_register_ground = xyz_register_ground[filter_mask, :]
            # self.xyz_register = self.xyz_register[filter_mask, :]
            # self.color_register = self.color_register[filter_mask, :]
            # rospy.loginfo("GridFilter: {} - > {}".format(filter_mask.shape[0], xyz_register_ground.shape[0]))

            pc_msg = xyz_array_to_point_cloud_msg(xyz_register_ground, pc_msg.header.stamp, pc_msg.header.frame_id,  self.color_register)
            self.cloud_pub_.publish(pc_msg)


        else:
            self.xyz_register = xyz_array
            self.color_register = color_array
            xyz_register_ground = transform_ground(self.xyz_register)


        self.localmap.pc_to_map(xyz_register_ground, self.color_register)

        heightmap = self.localmap.get_heightmap()
        grid_map_msg = self.to_gridmap_rosmsg(heightmap, pc_msg.header.stamp) # use the max height for now TODO
        grid_map_msg.info.header.stamp = pc_msg.header.stamp
        self.height_pub_.publish(grid_map_msg)

        rgbmap = self.localmap.get_rgbmap()
        # imgmsg = self.cvbridge.cv2_to_imgmsg(rgbmap, encoding="rgb8")

        imgmsg = Image()
        imgmsg.height = rgbmap.shape[0]
        imgmsg.width = rgbmap.shape[1]
        imgmsg.encoding = "rgb8"
        imgmsg.data = rgbmap.tostring()
        imgmsg.step = len(rgbmap.data)
        imgmsg.header.stamp = pc_msg.header.stamp

        self.rgb_pub_.publish(imgmsg)

        # if self.visualize_maps:
        #     self.localmap.show_heightmap()
        #     self.localmap.show_colormap()
        rospy.loginfo('[Mapping node] Receive {} points, after registration {}, time {}'.format(xyz_array.shape[0], self.xyz_register.shape[0], time.time()-starttime))
        # import ipdb;ipdb.set_trace()

if __name__ == '__main__':

    rospy.init_node("local_mapping", log_level=rospy.INFO)

    rospy.loginfo("local_mapping node initialized")
    node = LocalMappingRegisterNode()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if node.visualize_maps:
            node.localmap.show_heightmap()
            node.localmap.show_colormap()
        r.sleep()


        




