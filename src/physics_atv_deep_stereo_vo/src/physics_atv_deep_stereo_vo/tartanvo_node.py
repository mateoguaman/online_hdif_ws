#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
# from cv_bridge import CvBridge
import message_filters
from tf import TransformBroadcaster

import torch

from physics_atv_deep_stereo_vo.utils import ToTensor, Compose, CropCenter, Normalize, ResizeData, ColorCorrection
from physics_atv_deep_stereo_vo.utils import se2SE, SO2quat, se2quat
from physics_atv_deep_stereo_vo.TartanSVO import TartanSVO
import time
import os


def image_rectify(left_color, right_color, maps):
    limg = cv2.remap( left_color, maps[0], maps[1], cv2.INTER_LINEAR )
    rimg = cv2.remap( right_color, maps[2], maps[3], cv2.INTER_LINEAR )
    return limg, rimg

class TartanSVONodeBase(object):
    def __init__(self):

        self.w = rospy.get_param('~image_width', 1024)
        self.h = rospy.get_param('~image_height', 544)
        fx = rospy.get_param('~focal_x', 477.6049499511719)
        fy = rospy.get_param('~focal_y', 477.6049499511719)
        ox = rospy.get_param('~center_x', 499.5)
        oy = rospy.get_param('~center_y', 252.0)
        self.resize_w = rospy.get_param('~resize_w', 844)
        self.resize_h = rospy.get_param('~resize_h', 448)
        self.input_w = rospy.get_param('~input_w', 640)
        self.input_h = rospy.get_param('~input_h', 448)
        leftimgtopic = rospy.get_param('~left_image_topic', '/multisense/left/image_rect')
        rightimgtopic = rospy.get_param('~right_image_topic', '/multisense/right/image_rect')
        self.loophz = rospy.get_param('~main_loop_frequency', 100)
        self.blxfx = rospy.get_param('~focal_x_baseline', 100.14994812011719)
        self.color_correction = rospy.get_param('~color_correction', False)

        self.image_compressed = rospy.get_param('~image_compressed', False) # listern to compressed images
        self.stereo_maps = rospy.get_param('~stereo_maps', '') # listern to compressed images
        self.approximateSync = rospy.get_param('~approximate_sync', False) # use approximate sync instead of exact sync
        self.world_tf_name = rospy.get_param('~world_tf_name', '/warthog5/odom')  # On atv this is "multisense_init"
        self.base_tf_name = rospy.get_param('~base_tf_name', '/warthog5/base')  # On atv this is "multisense"

        self.cam_intrinsics = [self.w, self.h, fx, fy, ox, oy]
        self.curdir = os.path.dirname(os.path.realpath(__file__))

        # self.cv_bridge = CvBridge()

        self.intrinsic_np = self.make_intrinsics_layer(self.w, self.h, fx, fy, ox, oy)
        self.load_model()

        self.pose_pub = rospy.Publisher("tartanvo_pose", PoseStamped, queue_size=1)
        self.odom_pub = rospy.Publisher("tartanvo_odom", Odometry, queue_size=1)
        self.trans_pub = rospy.Publisher("tartanvo_transform", TransformStamped, queue_size=1)

        if self.stereo_maps != '':
            loadmap = np.load(self.curdir+'/'+self.stereo_maps, allow_pickle=True)
            loadmap = loadmap.item()
            # import ipdb;ipdb.set_trace()
            map1, map2 = cv2.initUndistortRectifyMap(\
                        loadmap['k1'], loadmap['d1'],\
                        loadmap['r1'], loadmap['p1'],\
                        (self.w, self.h), cv2.CV_32FC1)

            map3, map4 = cv2.initUndistortRectifyMap(\
                        loadmap['k2'], loadmap['d2'],\
                        loadmap['r2'], loadmap['p2'],\
                        (self.w, self.h), cv2.CV_32FC1)
            self.stereomaps = [map1, map2, map3, map4]

        if self.image_compressed:
            ImageType = CompressedImage
        else:
            ImageType = Image
        leftimg_sync = message_filters.Subscriber(leftimgtopic, ImageType)
        rightimg_sync = message_filters.Subscriber(rightimgtopic, ImageType)

        if self.approximateSync:
            ts = message_filters.ApproximateTimeSynchronizer([leftimg_sync, rightimg_sync], 200, 0.02, allow_headerless=False)
        else:
            ts = message_filters.TimeSynchronizer([leftimg_sync, rightimg_sync], 200)
        ts.registerCallback(self.handle_imgs)
        rospy.Subscriber('cam_info', CameraInfo, self.handle_caminfo)

        self.tf_broadcaster = TransformBroadcaster()

        # rospy.Subscriber('rgb_image', Image, self.handle_img)
        # rospy.Subscriber('vo_scale', Float32, self.handle_scale)

        self.last_left_img = None
        self.last_right_img = None
        self.pose = np.matrix(np.eye(4,4))
        self.imgbuf = []
        # self.scale = 1.0

    def make_intrinsics_layer(self, w, h, fx, fy, ox, oy):

        ww, hh = np.meshgrid(range(w), range(h))
        ww = (ww.astype(np.float32) - ox + 0.5 )/fx
        hh = (hh.astype(np.float32) - oy + 0.5 )/fy
        intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

        # handle resize - hard code
        intrinsicLayer = cv2.resize(intrinsicLayer,(self.resize_w, self.resize_h))
        # handle crop 
        cropsize_w = int((self.resize_w - self.input_w)/2)
        cropsize_h = int((self.resize_h - self.input_h)/2)
        intrinsicLayer = intrinsicLayer[cropsize_h:self.resize_h-cropsize_h, cropsize_w:self.resize_w-cropsize_w, :]
        # handle downsampleflow
        # intrinsicLayer = cv2.resize(intrinsicLayer,(160, 112))    
        # handle to_tensor
        intrinsicLayer = intrinsicLayer.transpose(2, 0, 1)
        intrinsicLayer = intrinsicLayer[None] # add one dimension

        return intrinsicLayer


    def load_model(self,):
        raise NotImplementedError
        
    def handle_caminfo(self, msg):
        # P: [477.6049499511719, 0.0, 499.5, -100.14994812011719, 0.0, 477.6049499511719, 252.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        w = msg.width
        h = msg.height
        fx = msg.P[0]
        fy = msg.P[5]
        ox = msg.P[2]
        oy = msg.P[6]
        new_intrinsics = [w, h, fx, fy, ox, oy]
        change = [xx!=yy for xx,yy in zip(new_intrinsics, self.cam_intrinsics)]
        if True in change:
            self.intrinsic_np = self.make_intrinsics_layer(w, h, fx, fy, ox, oy)
            self.cam_intrinsics = [w, h, fx, fy, ox, oy]
            self.blxfx = msg.P[3]
            print('Camera intrinsics updated..')

    def pub_motion(self, motion, stamp):
        motion_quat = se2quat(motion) # x, y, z, rx, ry, rz, rw
        trans_msg = TransformStamped()
        trans_msg.header.stamp = stamp
        trans_msg.header.frame_id = self.base_tf_name
        trans_msg.transform.translation.x = motion_quat[0]
        trans_msg.transform.translation.y = motion_quat[1]
        trans_msg.transform.translation.z = motion_quat[2]
        trans_msg.transform.rotation.x = motion_quat[3]
        trans_msg.transform.rotation.y = motion_quat[4]
        trans_msg.transform.rotation.z = motion_quat[5]
        trans_msg.transform.rotation.w = motion_quat[6]
        self.trans_pub.publish(trans_msg)


        motion_mat = se2SE(motion)
        self.pose = self.pose * motion_mat
        quat = SO2quat(self.pose[0:3,0:3])

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.world_tf_name
        pose_msg.pose.position.x = self.pose[0,3]
        pose_msg.pose.position.y = self.pose[1,3]
        pose_msg.pose.position.z = self.pose[2,3]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)     

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.pose.pose = pose_msg.pose

        self.odom_pub.publish(odom_msg)      

        # self.tf_broadcaster.sendTransform(
        #     (self.pose[0,3], self.pose[1,3], self.pose[2,3]),
        #     (quat[0], quat[1], quat[2], quat[3]),
        #     stamp, self.base_tf_name, self.world_tf_name)

    def handle_imgs(self, leftmsg, rightmsg):
        # import ipdb;ipdb.set_trace()
        # starttime = time.time()
        # image_left_np = self.cv_bridge.imgmsg_to_cv2(leftmsg, "bgr8")
        # image_right_np = self.cv_bridge.imgmsg_to_cv2(rightmsg, "bgr8")

        if not self.image_compressed:
            image_left_np = np.frombuffer(leftmsg.data, dtype=np.uint8).reshape(leftmsg.height, leftmsg.width, -1)
            image_right_np = np.frombuffer(rightmsg.data, dtype=np.uint8).reshape(rightmsg.height, rightmsg.width, -1)
        else:
            image_left_np = cv2.imdecode(np.frombuffer(leftmsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.h, self.w, -1)
            image_right_np = cv2.imdecode(np.frombuffer(rightmsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.h, self.w, -1)

        if self.stereo_maps != '': # rectify stereo images (for warthog)
            image_left_np, image_right_np = image_rectify(image_left_np, image_right_np, self.stereomaps)

        self.imgbuf.append([image_left_np, image_right_np, leftmsg.header.stamp])

    def main_loop(self):
        raise NotImplementedError

class TartanSVONode(TartanSVONodeBase):
    def __init__(self):
        super(TartanSVONode, self).__init__()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.color_correction:
            self.transform = Compose([ResizeData((self.resize_h, self.resize_w)), 
                                    CropCenter((self.input_h, self.input_w)), 
                                    ColorCorrection(),
                                    Normalize(mean=mean,std=std,keep_old=True), 
                                    ToTensor()]) 
        else:
            self.transform = Compose([ResizeData((self.resize_h, self.resize_w)), 
                                    CropCenter((self.input_h, self.input_w)), 
                                    Normalize(mean=mean,std=std,keep_old=True), 
                                    ToTensor()]) 

    def load_model(self,):
        model_name = rospy.get_param('~model_name', '43_6_2_vonet_30000_wo_pwc.pkl')
        network_type = rospy.get_param('~network_type', '2')
        print(f"Inside tartanvo_node. curdir is: {self.curdir}")
        model_name = self.curdir + '/models/' + model_name
        self.tartanvo = TartanSVO(model_name, network=network_type, blxfx=self.blxfx*self.resize_w/self.w)

    def main_loop(self):
        r = rospy.Rate(self.loophz)
        while not rospy.is_shutdown(): # loop just for visualization
            datalen = len(self.imgbuf)
            starttime = time.time()
            if datalen>0:
                starttime = time.time()
                if datalen>1:
                    rospy.logwarn("[VO node] Buffer len {}, skip {} frames..".format(datalen, datalen-1))
                image_left_np, image_right_np, time_stamp = self.imgbuf[-1]
                self.imgbuf = []

                if (image_left_np.shape[2]) == 1:
                    image_left_np = np.tile(image_left_np,(1,1,3))
                    image_right_np = np.tile(image_right_np,(1,1,3))

                if self.last_left_img is not None:
                    sample = {'img0': self.last_left_img, 
                            'img0n': image_left_np, 
                            'img1': self.last_right_img,
                            'blxfx': np.array([self.blxfx])
                            }
                    sample = self.transform(sample)
                    sample['img0'] = sample['img0'][None] # increase the dimension
                    sample['img0n'] = sample['img0n'][None]
                    sample['img1'] = sample['img1'][None]
                    sample['img0_norm'] = sample['img0_norm'][None]
                    sample['img0n_norm'] = sample['img0n_norm'][None]
                    sample['img1_norm'] = sample['img1_norm'][None]
                    sample['intrinsic'] = torch.from_numpy(self.intrinsic_np).float()

                    motion = self.tartanvo.test_batch(sample)
                    motion = motion[0]
                    # print(motion)
                    
                    self.pub_motion(motion, time_stamp)

                self.last_left_img = image_left_np.copy()
                self.last_right_img = image_right_np.copy()
                rospy.loginfo_throttle(1.0, "[VO node] vo inference time: {}:".format(time.time()-starttime))

            r.sleep()

if __name__ == '__main__':
    rospy.init_node("tartansvo_node", log_level=rospy.INFO)
    rospy.loginfo("tartanvo node initialized")
    node = TartanSVONode()
    node.main_loop()
