#!/usr/bin/env python

import numpy as np
import time

import rospy
import ros_numpy

from sensor_msgs.point_cloud2 import PointCloud2, PointField
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
import message_filters
# from cv_bridge import CvBridge
from nav_msgs.msg import Odometry

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ScrollGrid import ScrollGrid
from GridFilter import GridFilter

# import pcl

from utils import pointcloud2_to_xyzrgb_array, xyz_array_to_point_cloud_msg
from utils import pose2motion
from scipy.spatial.transform import Rotation

# def points_colors_to_pcl(points, colors):
#     '''
#     points: N x 4 float32
#     colors: N x 3 uint8
#     return pcl.PointCloud_PointXYZRGB
#     '''
#     pcl_pc = pcl.PointCloud_PointXYZRGB()
#     npts = points.shape[0]
#     pc_array = np.zeros((npts, 4), dtype=np.float32)
#     pc_array[:,:3] = points
#     colors_32 = colors.astype(np.uint32)
#     colors_combine = colors_32[:,0] << 16 | colors_32[:,1] << 8 | colors_32[:,2] 
#     # colors_combine = colors_32[:,0] << 16 | colors_32[:,1] << 8 | colors_32[:,2]
#     pc_array[:,3] = colors_combine.astype(np.float32)
#     pcl_pc.from_array(pc_array)
#     return pcl_pc

# def pcl_to_points_colors(pcl_pc):
#     pc_array = pcl_pc.to_array()
#     points = pc_array[:, :3]
#     colors_combine = pc_array[:, 3].astype(np.uint32)
#     npts = colors_combine.shape[0]
#     colors = np.zeros((npts, 3), dtype=np.uint8)
#     colors[:,0] = (colors_combine >> 16) % 256
#     colors[:,1] = (colors_combine >> 8) % 256
#     colors[:,2] = (colors_combine ) % 256

#     return points, colors

def coord_transform(points, R, T):
    points_trans = np.matmul(R.transpose(1,0), points.transpose(1,0)) - T
    return points_trans.transpose(1, 0)

# for debugging
def coord_transform2(points, R, T):
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    return points_trans.transpose(1, 0)

def optical2cam_transform(points):
    # the R and T come from gound calibration
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    # R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
    #               [ 0.,         -0.99969192, -0.02482067],
    #               [-0.24606434, 0.02405752,  -0.96895489]] )

    R0 = np.array([[0., 0., 1.],
                  [-1., 0., 0],
                  [0., -1., 0.]] )
    # manually adjust the camera-mocap extrinsics
    th1 = -0.0
    R1 = np.array([[np.cos(th1), 0., np.sin(th1)],
                   [0., 1., 0],
                   [-np.sin(th1), 0., np.cos(th1)]])
    th2 = -0.
    R2 = np.array([[np.cos(th2), -np.sin(th2), 0.],
                   [np.sin(th2), np.cos(th2), 0.],
                   [0., 0., 1]])
    # T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = points.transpose(1,0)
    points_trans = np.matmul(R0, points_trans)
    points_trans = np.matmul(R1, points_trans)
    points_trans = np.matmul(R2, points_trans)
    # points_trans = np.matmul(R, points_trans) + T
    return points_trans.transpose(1, 0)

def transform_ground(points):
    # the R and T come from gound calibration
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    return points_trans.transpose(1, 0)


class MushrPCPubFromBag(object):
    def __init__(self):
        import rosbag
        time.sleep(1)
        bagname = '/prague/arl_bag_files/mushr_bags/2021-08-13-16-08-06.bag' # mushr_x.bag' # 
        frame1 = 210
        frame2 = 220
        bag = rosbag.Bag(bagname, 'r')
        ind = 0
        self.cloud_pub_ = rospy.Publisher('deep_cloud_register', PointCloud2, queue_size=1)

        for topic, msg, t in bag.read_messages(topics=['/camera/depth/color/points']):
            if ind<frame1:
                ind += 1
                continue
            if ind==frame1:
                # self.cloud_pub_.publish(msg)
                # time.sleep(0.1)
                rospy.loginfo('Published first frame {}'.format(ind))
            import ipdb;ipdb.set_trace()
            self.cloud_pub_.publish(msg)
            rospy.loginfo('Published frame {}'.format(ind))
            time.sleep(1.1)
            if ind==frame2:
                # self.cloud_pub_.publish(msg)
                rospy.loginfo('Published second frame {}'.format(ind))
                break
            ind += 1



class MushrPointCloudPub(object):
    def __init__(self):

        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.handle_pc_static, queue_size=1) # /voxel_grid/output points need to be in NED
        rospy.Subscriber('/mushr_mocap/pose', PoseStamped, self.handle_pose, queue_size=200)

        self.last_pose = None
        self.recent_pose = None
        self.init_pose = None # for debugging
        self.odom_wait_num = 1
        self.odom_buffer = [[],] * self.odom_wait_num # manually adjust the sychronization issue
        self.odom_ind = 0

        self.xyz_register = None
        self.color_register = None

        self.pc_skip = 0
        self.pc_count = 0

        # self.cvbridge = CvBridge()
        self.gridfilter = GridFilter(0.01, (0., 3., -2., 2., 0, 1))

        # for debugging
        self.cloud_pub_ = rospy.Publisher('deep_cloud_register', PointCloud2, queue_size=1)


    def points_filter(self, points_3d, points, colors):
        # import ipdb;ipdb.set_trace()
        # points = points.reshape(-1, 3)
        mask = points_3d[:,0] > 1 #self.min_x - 2
        points_filter = points[mask, :]
        colors_filter = colors[mask, :]
        points_3d_filterd = points_3d[mask, :]
        print('Points filter: {} - {}'.format(points_3d.shape[0], points_3d_filterd.shape[0]))
        return points_3d_filterd, points_filter, colors_filter


    def handle_pose(self, msg):
        cur_pose = [msg.pose.position.x, 
                    msg.pose.position.y, 
                    msg.pose.position.z,
                    msg.pose.orientation.x, 
                    msg.pose.orientation.y, 
                    msg.pose.orientation.z, 
                    msg.pose.orientation.w]

        self.odom_buffer[self.odom_ind] = cur_pose
        self.odom_ind = (self.odom_ind + 1 ) % self.odom_wait_num
        # self.recent_pose = cur_pose
        if len(self.odom_buffer[self.odom_ind]) > 0:
            self.recent_pose = self.odom_buffer[self.odom_ind] # use old odom message because of the delay of the point cloud

        if self.init_pose is None:
            self.init_pose = cur_pose


    def handle_pc_static(self, pc_msg):
        if self.pc_count < self.pc_skip:
            self.pc_count = self.pc_count + 1
            return
        else:
            self.pc_count = 0

        xyz_array, color_array = pointcloud2_to_xyzrgb_array(pc_msg)
        xyz_array = optical2cam_transform(xyz_array)
        xyz_array, _, color_array = self.points_filter(xyz_array, xyz_array, color_array)

        if self.recent_pose is not None:
            motion = pose2motion(self.init_pose, self.recent_pose)
            recent_T = np.array([[motion[0]], 
                          [motion[1]],
                          [motion[2]]])
            quat = np.array([motion[3], 
                             motion[4],
                             motion[5],
                             motion[6]])

            recent_R = Rotation.from_quat(quat).as_dcm()

            xyz_array = coord_transform2(xyz_array, recent_R, recent_T)

            pc_msg = xyz_array_to_point_cloud_msg(xyz_array, pc_msg.header.stamp, frame_id='map', colorimg = color_array)
            self.cloud_pub_.publish(pc_msg)

            rospy.loginfo('Points {}'.format(xyz_array.shape[0]))


    def handle_pc(self, pc_msg):

        xyz_array, color_array = pointcloud2_to_xyzrgb_array(pc_msg)
        xyz_array = optical2cam_transform(xyz_array)

        if self.last_pose is not None:
            motion = pose2motion(self.last_pose, self.recent_pose)
            recent_T = np.array([[motion[0]], 
                          [motion[1]],
                          [motion[2]]])
            quat = np.array([motion[3], 
                             motion[4],
                             motion[5],
                             motion[6]])
            recent_R = Rotation.from_quat(quat).as_dcm()
        else:
            recent_R, recent_T = None, None

        self.last_pose = self.recent_pose

        if self.xyz_register is not None and recent_R is not None:
            points_trans = coord_transform(self.xyz_register, recent_R, recent_T)
            self.xyz_register = np.concatenate((xyz_array, points_trans),axis=0)
            self.color_register = np.concatenate((color_array, self.color_register),axis=0)
            # xyz_register_ground = transform_ground(self.xyz_register)

            # xyz_register_ground, self.xyz_register, self.color_register = self.points_filter(xyz_register_ground, self.xyz_register, self.color_register)
            # TODO filter by density
            filter_mask = self.gridfilter.grid_filter(self.xyz_register)
            self.xyz_register = self.xyz_register[filter_mask,:]
            self.color_register = self.color_register[filter_mask,:]
            pc_msg = xyz_array_to_point_cloud_msg(self.xyz_register, pc_msg.header.stamp, 'mushr', self.color_register)
            self.cloud_pub_.publish(pc_msg)


        else:
            self.xyz_register = xyz_array
            self.color_register = color_array
            xyz_register_ground = transform_ground(self.xyz_register)


        # self.localmap.pc_to_map(xyz_register_ground, self.color_register)

        # heightmap = self.localmap.get_heightmap()
        # grid_map_msg = self.to_gridmap_rosmsg(heightmap, pc_msg.header.stamp) # use the max height for now TODO
        # self.height_pub_.publish(grid_map_msg)
        # rgbmap = self.localmap.get_rgbmap()
        # imgmsg = self.cvbridge.cv2_to_imgmsg(rgbmap, encoding="rgb8")
        # self.rgb_pub_.publish(imgmsg)

        # if self.visualize_maps:
        #     self.localmap.show_heightmap()
        #     self.localmap.show_colormap()
        print('Receive {} points, after registration {}'.format(xyz_array.shape[0], self.xyz_register.shape[0]))
        # # import ipdb;ipdb.set_trace()

if __name__ == '__main__':

    rospy.init_node("mushr_pc_pub", log_level=rospy.INFO)
    rospy.loginfo("mushr_pc_pub node initialized")

    node = MushrPointCloudPub()
    rospy.spin()

    # node = MushrPCPubFromBag()

        




