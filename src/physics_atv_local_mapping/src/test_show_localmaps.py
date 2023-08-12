#!/usr/bin/env python
import cv2
import numpy as np

import rospy
import ros_numpy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header

import time


class TestLocalmap(object):
    def __init__(self):

        self.show_inflate = True
        rospy.Subscriber('/local_height_map', Image, self.handle_height, queue_size=1)
        rospy.Subscriber('/local_rgb_map', Image, self.handle_rgb, queue_size=1)

        self.cvbridge = CvBridge()
        self.heightmap = None
        self.rgbmap = None

        if self.show_inflate:
            rospy.Subscriber('/local_height_map_inflate', Image, self.handle_height_inflate, queue_size=1)
            rospy.Subscriber('/local_rgb_map_inflate', Image, self.handle_rgb_inflate, queue_size=1)
            self.heightmap_inflate = None
            self.rgbmap_inflate = None

    def handle_height(self, msg):
        self.heightmap = self.cvbridge.imgmsg_to_cv2(msg, "32FC4")
        print('Receive heightmap {}'.format(self.heightmap.shape))
        # import ipdb;ipdb.set_trace()

    def handle_rgb(self, msg):
        self.rgbmap = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        print('Receive rgbmap {}'.format(self.rgbmap.shape))

    def handle_height_inflate(self, msg):
        self.heightmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "32FC4")
        print('Receive heightmap {}'.format(self.heightmap_inflate.shape))
        # import ipdb;ipdb.set_trace()

    def handle_rgb_inflate(self, msg):
        self.rgbmap_inflate = self.cvbridge.imgmsg_to_cv2(msg, "rgb8")
        print('Receive rgbmap {}'.format(self.rgbmap_inflate.shape))

    def show_heightmap_base(self, dispmap, winname='height', hmin=-1.5, hmax=4, scale=0.5):
        if dispmap is None:
            return
        mask = dispmap[:,:,0]>1000
        disp1 = np.clip((dispmap[:, :, 0] - hmin)*100, 0, 255).astype(np.uint8)
        disp2 = np.clip((dispmap[:, :, 1] - hmin)*100, 0, 255).astype(np.uint8)
        disp3 = np.clip((dispmap[:, :, 2] - hmin)*100, 0, 255).astype(np.uint8)
        disp4 = np.clip(dispmap[:, :, 3]*1000, 0, 255).astype(np.uint8)
        disp1[mask] = 0
        disp2[mask] = 0
        disp3[mask] = 0
        disp4[mask] = 0
        dispc1 = np.concatenate((cv2.flip(disp1, -1), cv2.flip(disp2, -1)) , axis=1)
        dispc2 = np.concatenate((cv2.flip(disp3, -1), cv2.flip(disp4, -1)) , axis=1)
        disp = np.concatenate((dispc1, dispc2), axis=0)
        disp = cv2.resize(disp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        cv2.imshow(winname,disp_color)
        cv2.waitKey(1)

    def show_colormap_base(self, dispmap, winname='color', scale=1.5, bgr=False):
        if dispmap is None:
            return
        disp = cv2.flip(dispmap, -1)
        disp = cv2.resize(disp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if bgr:
            disp = disp[:,:,::-1]
        cv2.imshow(winname,disp)
        cv2.waitKey(1)

    def show_heightmap(self):
        self.show_heightmap_base(self.heightmap)
        if self.show_inflate:
            self.show_heightmap_base(self.heightmap_inflate, 'height_inflate')

    def show_colormap(self):
        self.show_colormap_base(self.rgbmap, bgr=True)
        if self.show_inflate:
            self.show_colormap_base(self.rgbmap_inflate, 'color_inflate', bgr=True)

if __name__ == '__main__':

    rospy.init_node("test_localmap", log_level=rospy.INFO)

    rospy.loginfo("test_localmap node initialized")
    node = TestLocalmap()
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): # loop just for visualization
        node.show_heightmap()
        node.show_colormap()


        




        # model_name = rospy.get_param('~model_name', '43_6_2_vonet_30000.pkl')
        # w = rospy.get_param('~image_width', 1024)
        # h = rospy.get_param('~image_height', 544)
        # fx = rospy.get_param('~focal_x', 477.6049499511719)
        # fy = rospy.get_param('~focal_y', 477.6049499511719)
        # ox = rospy.get_param('~center_x', 499.5)
        # oy = rospy.get_param('~center_y', 252.0)
        # self.cam_intrinsics = [w, h, fx, fy, ox, oy]
        # self.blxfx = 100.14994812011719

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # self.cv_bridge = CvBridge()
        # self.transform = Compose([ResizeData((448,843)), 
        #                           CropCenter((448, 640)), 
        #                           DownscaleFlow(), 
        #                           Normalize(mean=mean,std=std,keep_old=True), 
        #                           ToTensor()]) 
        # self.intrinsic = make_intrinsics_layer(w, h, fx, fy, ox, oy)
        # self.tartanvo = TartanSVO(model_name)

        # self.map_pub = rospy.Publisher("local_heightmap", Image, queue_size=10)

        # pc_sync = message_filters.Subscriber('/deep_cloud', Image)
        # odom_sync = message_filters.Subscriber('/tartanvo_pose', Image)
        # ts = message_filters.ApproximateTimeSynchronizer([leftimg_sync, rightimg_sync], 1, 0.01)
        # ts.registerCallback(self.handle_imgs)
        # rospy.Subscriber('cam_info', CameraInfo, self.handle_caminfo)

        # self.tf_broadcaster = TransformBroadcaster()

        # # rospy.Subscriber('rgb_image', Image, self.handle_img)
        # # rospy.Subscriber('vo_scale', Float32, self.handle_scale)

        # self.last_left_img = None
        # self.last_right_img = None
        # self.pose = np.matrix(np.eye(4,4))
        # # self.scale = 1.0
