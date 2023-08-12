#!/usr/bin/env python3

import cv2
import torch
import numpy as np

import rospy
from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge
import message_filters

# from PSM import stackhourglass as StereoNet
import time
import os

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from math import pi, tan
from physics_atv_deep_stereo_vo.utils import ColorCorrection

def image_warp(left_color, size):
    '''
    convert the left_color to left_color_rect
    it should not be necessary if we record the left_color_rect directly
    '''
    D = np.array([-0.03597196191549301, 0.08101943135261536, -0.00028405245393514633, 0.0011874435003846884, -0.03709280863404274, 0.0, 0.0, 0.0])
    K = np.array([455.77496337890625, 0.0, 497.1180114746094, 0.0, 456.319091796875, 251.58502197265625, 0.0, 0.0, 1.0]).reshape(3,3)
    R = np.array([0.9999926686286926, -0.001692004851065576, 0.003427336923778057, 0.0016923071816563606, 0.9999985694885254, -8.531315688742325e-05, -0.003427187679335475, 9.11126408027485e-05, 0.9999940991401672]).reshape(3,3)
    P = np.array([477.6049499511719, 0.0, 499.5, 0.0, 0.0, 477.6049499511719, 252.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4)

    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, size=size, m1type=cv2.CV_32FC1) 
    left_color_rect = cv2.remap(left_color, map1, map2, cv2.INTER_LINEAR) 
    return left_color_rect

def image_rectify(left_color, right_color, maps):
    limg = cv2.remap( left_color, maps[0], maps[1], cv2.INTER_LINEAR )
    rimg = cv2.remap( right_color, maps[2], maps[3], cv2.INTER_LINEAR )
    return limg, rimg


def depth_to_point_cloud(depth, focalx, focaly, pu, pv, filtermin=-1, filtermax=-1, colorimg=None, mask=None):
    """
    Convert depth image to point cloud based on intrinsic parameters
    :param depth: depth image
    :colorimg: a colored image that aligns with the depth, if not None, will return the color
    :mask: h x w bool array, throw away the points if mask value is false
    :return: xyz point array
    """
    h, w = depth.shape
    depth64 = depth.astype(np.float64)
    wIdx = np.linspace(0, w - 1, w, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    hIdx = np.linspace(0, h - 1, h, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    u, v = np.meshgrid(wIdx, hIdx)

    if filtermax!=-1:
        maskf = depth<filtermax
        if filtermin!=-1:
            maskf = np.logical_and(maskf, depth>filtermin)
    else:
        if filtermin!=-1:
            maskf = depth>filtermin

    if mask is not None:
        if maskf is not None:
            maskf = np.logical_and(maskf, mask)
        else:
            maskf = mask

    if maskf is not None:
        depth64 = depth64[maskf]
        depth = depth[maskf]
        if colorimg is not None:
            colorimg = colorimg[maskf]
        u = u[maskf]
        v = v[maskf]
    # print('Depth mask {} -> {}'.format(h*w, mask.sum()))

    x = (u - pu) * depth64 / focalx
    y = (v - pv) * depth64 / focaly
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    points = np.stack([depth, x, y], axis=1) #rotate_points(depth, x, y, mode) # amigo: this is in NED coordinate
    return points, colorimg

def mask_depth_std(depth, std_thresh = 1):
    """
    return a mask of the depth where the local std is smaller than std_thresh
    this will filter out points at the edge of an object 
    """
    # import ipdb;ipdb.set_trace()
    h, w = depth.shape
    depth_padding = np.pad(depth, ((1,1),(1,1)), 'edge') #np.zeros((h+2,w+2), dtype=np.float32)
    shiftind = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    depth_shift = [depth_padding[i:i+h,j:j+w] for (i,j) in shiftind]
    depth_sum9 = np.stack(depth_shift, axis=2)
    depth_std  = depth_sum9.std(axis=2)
    # cv2.imshow("test", (np.clip(depth_std, 0, 1) * 255).astype(np.uint8))
    # cv2.waitKey(0)
    mask = depth_std < std_thresh
    return mask

def coord_transform(points):
    # the R and T come from gound calibration
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    return points_trans.transpose(1, 0)

def points_height_filter(points, maxhight, colorimg=None):
    # import ipdb;ipdb.set_trace()
    # points = points.reshape(-1, 3)
    mask = points[:,2]<maxhight
    points_filter = points[mask, :]
    if colorimg is not None:
        colorimg_filter = colorimg[mask, :]
    else:
        colorimg_filter = None
    return points_filter, colorimg_filter


def xyz_array_to_point_cloud_msg(points, timestamp=None, colorimg=None, robot=None):
    """
    Please refer to this ros answer about the usage of point cloud message:
        https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
    :param points:
    :param header:
    :return:
    """
    header = Header()
    header.frame_id = f'{robot}/base'#  f'{robot}/stereo_left_link_ned'
    if timestamp is None:
        timestamp = rospy.Time().now()
    header.stamp = timestamp
    msg = PointCloud2()
    msg.header = header
    if len(points.shape)==3:
        msg.width = points.shape[0]
        msg.height = points.shape[1]
    else:
        msg.width = points.shape[0]
        msg.height = 1
    msg.is_bigendian = False
    # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
    # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
    msg.is_dense = False

    if colorimg is None:
        msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        xyz = points.astype(np.float32)
        msg.data = xyz.tostring()
    else:
        msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), 
                      PointField('rgb', 12, PointField.UINT32, 1),]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        xyzcolor = np.zeros( (points.shape[0], 1), \
        dtype={ 
            "names": ( "x", "y", "z", "rgba" ), 
            "formats": ( "f4", "f4", "f4", "u4" )} )
        xyzcolor["x"] = points[:, 0].reshape((-1, 1))
        xyzcolor["y"] = points[:, 1].reshape((-1, 1))
        xyzcolor["z"] = points[:, 2].reshape((-1, 1))
        color_rgba = np.zeros((points.shape[0], 4), dtype=np.uint8) + 255
        color_rgba[:,:3] = colorimg
        xyzcolor["rgba"] = color_rgba.view('uint32')
        msg.data = xyzcolor.tostring()

    return msg

class StereoNodeBase(object):
    def __init__(self, ):
        self.curdir = os.path.dirname(os.path.realpath(__file__))

        self.load_model() # load model from file
        self.disp = None
        self.databuf = []

        self.cloud_pub_ = rospy.Publisher('deep_cloud', PointCloud2, queue_size=1)

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        # camera parameters
        self.w = rospy.get_param('~image_width', 1024)
        self.h = rospy.get_param('~image_height', 544)
        self.img_w, self.img_h = self.w, self.h
        self.focalx = rospy.get_param('~focal_x', 477.6049499511719)
        self.focaly = rospy.get_param('~focal_y', 477.6049499511719)
        self.pu = rospy.get_param('~center_x', 499.5)
        self.pv = rospy.get_param('~center_y', 252.0)
        self.fxbl = rospy.get_param('~focal_x_baseline', 100.14994812011719)

        # depth generation parameters
        self.crop_w = rospy.get_param('~image_crop_w', 64) # to deal with vignette effect, crop the image
        self.crop_h_low = rospy.get_param('~image_crop_h_low', 32) # after cropping the size is (960, 512)
        self.crop_h_high = rospy.get_param('~image_crop_h_high', 32) # crop the top part of the image
        self.input_w = rospy.get_param('~image_input_w', 512)
        self.input_h = rospy.get_param('~image_input_h', 256)
        self.visualize = rospy.get_param('~visualize_depth', True)
        self.filter_outlier = rospy.get_param('~filter_outlier', True)
        self.uncertainty_thresh = rospy.get_param('~uncertainty_thresh', -2.5) # only publish points with low uncertainty

        # point cloud processing parameters
        self.mindist = rospy.get_param('~pc_min_dist', 1.0) # not filter if set to -1 
        self.maxdist = rospy.get_param('~pc_max_dist', 15.0) # not filter if set to -1
        self.maxhight = rospy.get_param('~pc_max_height', 2.0) # not filter if set to -1
        self.loophz = rospy.get_param('~main_loop_frequency', 100)

        self.leftimgtopic = rospy.get_param('~left_image_topic', '/multisense/left/image_rect')
        self.rightimgtopic = rospy.get_param('~right_image_topic', '/multisense/right/image_rect')
        self.color_topic = rospy.get_param('~color_image_topic', '/multisense/left/image_rect_color')
        self.maskfile = rospy.get_param('~mask_file', 'atvmask.npy')
        self.image_rect = rospy.get_param('~image_rect', True) # listern to rectified colored image
        self.image_compressed = rospy.get_param('~image_compressed', False) # listern to compressed images
        self.stereo_maps = rospy.get_param('~stereo_maps', '') # do stereo rectification
        self.approximateSync = rospy.get_param('~approximate_sync', False) # use approximate sync instead of exact sync
        self.robot = rospy.get_param('~robot', 'warthog5')
        self.color_correction = rospy.get_param('~color_correction', False)

        # some flags to control the point cloud processing
        self.transform_ground = rospy.get_param('~pc_transform_ground', False)
        self.crop_intrinsics()
        self.scale_intrinsics()
        self.atvmask = np.load(self.curdir+'/'+self.maskfile)
        self.crop_resize_atvmask()
        self.atvmask = self.atvmask < 10 # a threshold

        if self.stereo_maps != '':
            loadmap = np.load(self.curdir+'/'+self.stereo_maps, allow_pickle=True)
            print(f"Loading this stereo map: {self.curdir+'/'+self.stereo_maps}")
            loadmap = loadmap.item()
            # import ipdb;ipdb.set_trace()
            map1, map2 = cv2.initUndistortRectifyMap(\
                        loadmap['k1'], loadmap['d1'],\
                        loadmap['r1'], loadmap['p1'],\
                        (self.img_w, self.img_h), cv2.CV_32FC1)
            map3, map4 = cv2.initUndistortRectifyMap(\
                        loadmap['k2'], loadmap['d2'],\
                        loadmap['r2'], loadmap['p2'],\
                        (self.img_w, self.img_h), cv2.CV_32FC1)
            self.stereomaps = [map1, map2, map3, map4]

        if self.color_correction:
            self.color_corrector = ColorCorrection()

        if self.image_compressed:
            ImageType = CompressedImage
        else:
            ImageType = Image
        leftimg_sync = message_filters.Subscriber(self.leftimgtopic, ImageType)
        rightimg_sync = message_filters.Subscriber(self.rightimgtopic, ImageType)
        if self.color_topic == "": # use left ImageType as the source of rgb texture
            if self.approximateSync:
                ts = message_filters.ApproximateTimeSynchronizer([leftimg_sync, rightimg_sync], 200, 0.02, allow_headerless=False)
            else:
                ts = message_filters.TimeSynchronizer([leftimg_sync, rightimg_sync], 200)
            ts.registerCallback(self.handle_imgs)
        else:
            colorimg_sync = message_filters.Subscriber(self.color_topic, ImageType)
            if self.approximateSync:
                ts = message_filters.ApproximateTimeSynchronizer([leftimg_sync, rightimg_sync, colorimg_sync], 200, 0.02, allow_headerless=False)
            else:
                ts = message_filters.TimeSynchronizer([leftimg_sync, rightimg_sync, colorimg_sync], 10)
            ts.registerCallback(self.handle_imgs_with_color)

        # if self.stereo_maps != '':
        #     loadmap = np.load(self.curdir+'/'+self.stereo_maps, allow_pickle=True)
        #     loadmap = loadmap.item()
        #     # import ipdb;ipdb.set_trace()
        #     map1, map2 = cv2.initUndistortRectifyMap(\
        #                 loadmap['k1'], loadmap['d1'],\
        #                 loadmap['r1'], loadmap['p1'],\
        #                 (self.img_w, self.img_h), cv2.CV_32FC1)

        #     map3, map4 = cv2.initUndistortRectifyMap(\
        #                 loadmap['k2'], loadmap['d2'],\
        #                 loadmap['r2'], loadmap['p2'],\
        #                 (self.img_w, self.img_h), cv2.CV_32FC1)
        #     self.stereomaps = [map1, map2, map3, map4]

    def load_model(self, ):
        raise NotImplementedError

    def scale_imgs(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        leftImg = cv2.resize(leftImg,(self.input_w, self.input_h))
        rightImg = cv2.resize(rightImg,(self.input_w, self.input_h))

        return {'left': leftImg, 'right': rightImg}

    def scale_intrinsics(self):
        scalex = float(self.input_w)/self.w
        scaley = float(self.input_h)/self.h
        self.focalx = self.focalx * scalex
        self.focaly = self.focaly * scaley
        self.pu = self.pu * scalex
        self.pv = self.pv * scaley
        self.fxbl = self.fxbl * scalex

    def crop_imgs(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        h, w, c = leftImg.shape
        leftImg = leftImg[self.crop_h_low:h-self.crop_h_high, self.crop_w:w-self.crop_w, :]
        rightImg = rightImg[self.crop_h_low:h-self.crop_h_high, self.crop_w:w-self.crop_w, :]

        return {'left': leftImg, 'right': rightImg}

    def crop_intrinsics(self):
        self.pu = self.pu - self.crop_w
        self.pv = self.pv - self.crop_h_low
        self.w = self.w - 2 * self.crop_w
        self.h = self.h - self.crop_h_low - self.crop_h_high

    def crop_resize_atvmask(self):
        h, w = self.atvmask.shape
        self.atvmask = self.atvmask[self.crop_h_low:h-self.crop_h_high, self.crop_w:w-self.crop_w]
        self.atvmask = cv2.resize(self.atvmask,(self.input_w, self.input_h))

    # def scale_back(self, disparity):
    #     disparity = cv2.resize(disparity, (self.w, self.h))
    #     disparity = disparity * self.w / self.input_w
    #     return disparity

    def to_tensor(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        leftImg = leftImg.transpose(2,0,1)/float(255)
        rightImg = rightImg.transpose(2,0,1)/float(255)
        leftTensor = torch.from_numpy(leftImg).float()
        rightTensor = torch.from_numpy(rightImg).float()
        # rgbsTensor = torch.cat((leftTensor, rightTensor), dim=0)
        return {'left': leftTensor, 'right': rightTensor}


    def normalize(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        for t, m, s in zip(leftImg, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(rightImg, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'left': leftImg, 'right': rightImg}

    def disp2vis(self, disp, scale=10,):
        '''
        disp: h x w float32 numpy array
        return h x w x 3 uint8 numpy array
        '''
        disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        # disp = cv2.resize(disp,(640,480))
        return disp_color

    def visualize_depth(self, disp, output_unc, left_image_np, right_image_np):
        disp1 = np.concatenate((left_image_np, right_image_np), axis=1)
        factor = (disp.shape[1]*2)/disp1.shape[1]
        disp1 = cv2.resize(disp1, (0,0), fx=factor, fy=factor)
        for k in range(10,disp1.shape[0],20):
            disp1[k,:,:] = [0,255,255]
        # import ipdb;ipdb.set_trace()
        # if disp is not None:
        dispvis = self.disp2vis(disp, scale=3)
        if output_unc is not None:
            dispunc = np.clip((output_unc + 4)*50, 0, 255).astype(np.uint8)
            dispunc = np.tile(dispunc[...,np.newaxis], (1,1,3))
        else:
            dispunc = np.zeros_like(dispvis)
        disp2 = np.concatenate((dispvis, dispunc), axis=1)
        cv2.imshow('img', np.concatenate((disp1, disp2), axis=0))
        cv2.waitKey(1)

    # TODO: visualize std mask
    def pub_pc2(self, disparity, timestamp, color_image_warp, uncertainty):
        # import ipdb;ipdb.set_trace()
        depth = self.fxbl / (disparity + 1e-6)
        if self.filter_outlier:
            mask = mask_depth_std(depth, std_thresh = 0.3)
            # cv2.imshow('mask',mask.astype(np.uint8)*255)
            # cv2.waitKey(1)
            mask = np.logical_and(mask, self.atvmask)
        else:
            mask = self.atvmask

        if uncertainty is not None:
            mask_unc = uncertainty < self.uncertainty_thresh
            mask = np.logical_and(mask, mask_unc)

        # point_array = depth_to_point_cloud(depth, self.focalx, self.focaly, self.pu, self.pv, self.mindist, self.maxdist)
        point_array, color_array = depth_to_point_cloud(depth, self.focalx, self.focaly, self.pu, self.pv, 
                                                        self.mindist, self.maxdist, color_image_warp, mask)
        if self.transform_ground:
            point_array = coord_transform(point_array)
            if self.maxhight != -1:
                point_array, color_array = points_height_filter(point_array, self.maxhight, color_array)
        pc_msg = xyz_array_to_point_cloud_msg(point_array, timestamp, color_array, robot=self.robot)
        self.cloud_pub_.publish(pc_msg)

    def network_inference(self, sample):
        raise NotImplementedError

    def imgs_processing(self, left_image_np, right_image_np, color_image_np=None):
        # import ipdb;ipdb.set_trace()
        if (left_image_np.shape[2]) == 1:
            left_image_np = np.tile(left_image_np,(1,1,3))
            right_image_np = np.tile(right_image_np,(1,1,3))
        # print(left_image_np.shape, right_image_np.shape, color_image_np.shape)
        sample = {'left': left_image_np, 'right': right_image_np}
        # crop the image due to the vignette effect
        sample = self.crop_imgs(sample)
        if color_image_np is not None:
            h,w,_ = color_image_np.shape
            if not self.image_rect:
                color_image_np = image_warp(color_image_np, size=(self.img_w, self.img_h))
            color_image_np = color_image_np[self.crop_h_low:h-self.crop_h_high, self.crop_w:w-self.crop_w, :]

        if self.w != self.input_w or self.h != self.input_h:
            sample = self.scale_imgs(sample)
            if color_image_np is not None:
                color_image_np = cv2.resize(color_image_np, (self.input_w, self.input_h))

        if self.color_correction:
            sample = self.color_corrector(sample)

            if color_image_np is not None:
                color_image_np = self.color_correction.correct_color(color_image_np)

        return sample, color_image_np

    def handle_imgs(self, leftmsg, rightmsg):
        # print 'img received..'
        # rospy.loginfo('[Stereo node] Callback called with time stamp {}, {}..'.format(leftmsg.header.stamp.secs, leftmsg.header.stamp.nsecs))
        # starttime = time.time()
        # left_image_np = self.cv_bridge.imgmsg_to_cv2(leftmsg, "bgr8")
        # right_image_np = self.cv_bridge.imgmsg_to_cv2(rightmsg, "bgr8") 
        # color_image_np = self.cv_bridge.imgmsg_to_cv2(colormsg, "bgr8") 
        # import ipdb;ipdb.set_trace()
        if not self.image_compressed:
            left_image_np = np.frombuffer(leftmsg.data, dtype=np.uint8).reshape(leftmsg.height, leftmsg.width, -1)
            right_image_np = np.frombuffer(rightmsg.data, dtype=np.uint8).reshape(rightmsg.height, rightmsg.width, -1)
        else:
            left_image_np = cv2.imdecode(np.frombuffer(leftmsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.img_h, self.img_w, -1)
            right_image_np = cv2.imdecode(np.frombuffer(rightmsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.img_h, self.img_w, -1)

        self.databuf.append([left_image_np, right_image_np, None, leftmsg.header.stamp]) # return None for color image

    def handle_imgs_with_color(self, leftmsg, rightmsg, colormsg):
        if not self.image_compressed:
            left_image_np = np.frombuffer(leftmsg.data, dtype=np.uint8).reshape(leftmsg.height, leftmsg.width, -1)
            right_image_np = np.frombuffer(rightmsg.data, dtype=np.uint8).reshape(rightmsg.height, rightmsg.width, -1)
            color_image_np = np.frombuffer(colormsg.data, dtype=np.uint8).reshape(colormsg.height, colormsg.width, -1)
        else:
            left_image_np = cv2.imdecode(np.frombuffer(leftmsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.img_h, self.img_w, -1)
            right_image_np = cv2.imdecode(np.frombuffer(rightmsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.img_h, self.img_w, -1)
            color_image_np = cv2.imdecode(np.frombuffer(colormsg.data, np.uint8), cv2.IMREAD_COLOR).reshape(self.img_h, self.img_w, -1)

        self.databuf.append([left_image_np, right_image_np, color_image_np, leftmsg.header.stamp])

    def main_loop(self):
        raise NotImplementedError

class StereoNode(StereoNodeBase):
    def __init__(self):
        super(StereoNode, self).__init__()

    def load_model(self, ):
        '''
        load model into self.stereonet
        '''
        model_name = rospy.get_param('~model_name', '5_5_4_stereo_30000.pkl')
        self.network_type = rospy.get_param('~network_type', 0)
        if self.network_type == 0:
            from physics_atv_deep_stereo_vo.StereoNet7 import StereoNet7 as StereoNet
        elif self.network_type == 1:
            from physics_atv_deep_stereo_vo.StereoFlowNet import StereoNet
        elif self.network_type == 2:
            from physics_atv_deep_stereo_vo.PSM import stackhourglass as StereoNet

        # modelname = '../models/4_3_3_stereo_60000.pkl'
        print(f"Inside stereo_node_multisense. curdir is: {self.curdir}")
        modelname = self.curdir + '/models/' + model_name
        self.stereonet = StereoNet()

        preTrainDict = torch.load(modelname)
        model_dict = self.stereonet.state_dict()
        # print 'preTrainDict:',preTrainDict.keys()
        # print 'modelDict:',model_dict.keys()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            # self.logger.info("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]

                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

            preTrainDict = preTrainDictTemp

        # if ( 0 == len(preTrainDict) ):
        #     raise WorkFlow.WFException("Could not load stereonet from %s." % (modelname), "load_model")

        # for item in preTrainDict:
        #     print("Load pretrained layer:{}".format(item) )
        model_dict.update(preTrainDict)
        self.stereonet.load_state_dict(model_dict)

        self.stereonet.cuda()
        self.stereonet.eval()
        print('Stereo Model Loaded...')

    def to_tensor(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        leftImg = leftImg.transpose(2,0,1)/float(255)
        rightImg = rightImg.transpose(2,0,1)/float(255)
        leftTensor = torch.from_numpy(leftImg).float()
        rightTensor = torch.from_numpy(rightImg).float()
        # rgbsTensor = torch.cat((leftTensor, rightTensor), dim=0)
        return {'left': leftTensor, 'right': rightTensor}


    def normalize(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        for t, m, s in zip(leftImg, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(rightImg, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'left': leftImg, 'right': rightImg}

    def main_loop(self,):
        '''
        Can not do TensorRT inference in the callback function
        The walk around is to buffer the data in the callback, and process it in the main loop
        '''
        r = rospy.Rate(self.loophz)

        while not rospy.is_shutdown(): # loop just for visualization
            datalen = len(self.databuf)
            if datalen>0:
                starttime = time.time()
                if datalen>1:
                    rospy.logwarn("[Stereo node] Buffer len {}, skip {} frames..".format(datalen, datalen-1))
                left_image_np, right_image_np, color_image_np, time_stamp = self.databuf[-1]
                self.databuf = []
                if self.stereo_maps != '': # rectify stereo images (for warthog)
                    left_image_np, right_image_np = image_rectify(left_image_np, right_image_np, self.stereomaps)
                sample, color_image_np = self.imgs_processing(left_image_np, right_image_np, color_image_np)
                if color_image_np is None:
                    color_image_np = sample['left'].copy()
                sample = self.to_tensor(sample)
                sample = self.normalize(sample) 

                with torch.no_grad():
                    left_dims = sample['left'].shape
                    right_dims = sample['right'].shape
                    leftTensor = sample['left'].unsqueeze(0).cuda()
                    rightTensor = sample['right'].unsqueeze(0).cuda()
                    inputTensor = torch.cat((leftTensor, rightTensor), dim=1)
                    output, output_unc = self.stereonet(inputTensor)
                    if output_unc is not None:
                        output_unc = output_unc.detach().cpu().squeeze().numpy()

                disp = output.cpu().squeeze().numpy() 
                if self.network_type ==0 or self.network_type == 1:
                    disp = disp * 50 # 50 is the stereo normalization parameter

                self.pub_pc2(disp, time_stamp, color_image_np, output_unc)
                rospy.loginfo_throttle(1.0, '[Stereo node] Inference time {}'.format(time.time()-starttime))
                if self.visualize:
                    self.visualize_depth(disp, output_unc, left_image_np, right_image_np)
            r.sleep()


if __name__ == '__main__':

    rospy.init_node("stereo_net", log_level=rospy.INFO)
    rospy.loginfo("stereo_net_node initialized")

    node = StereoNode()
    node.main_loop()

    # # test image warp
    # inputimg = '/prague/arl_bag_files/data_collection_grass_3_2021_0524_calibration/left_color/1621879779527275000_left_color.png'
    # targetimg = '/prague/arl_bag_files/data_collection_grass_3_2021_0524_calibration/left_rect/1621879779527275000_left_mono.png'
    # left_color = cv2.imread(inputimg)
    # left_rect = cv2.imread(targetimg)
    # D = np.array([-0.03597196191549301, 0.08101943135261536, -0.00028405245393514633, 0.0011874435003846884, -0.03709280863404274, 0.0, 0.0, 0.0])
    # K = np.array([455.77496337890625, 0.0, 497.1180114746094, 0.0, 456.319091796875, 251.58502197265625, 0.0, 0.0, 1.0]).reshape(3,3)
    # R = np.array([0.9999926686286926, -0.001692004851065576, 0.003427336923778057, 0.0016923071816563606, 0.9999985694885254, -8.531315688742325e-05, -0.003427187679335475, 9.11126408027485e-05, 0.9999940991401672]).reshape(3,3)
    # P = np.array([477.6049499511719, 0.0, 499.5, 0.0, 0.0, 477.6049499511719, 252.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4)
    # # import ipdb;ipdb.set_trace()
    # left_rect_warp = image_warp(left_color, K, D, R, P)
    # disp1 = np.concatenate((left_color, left_rect_warp), axis=0)
    # disp2 = np.concatenate((left_rect_warp, left_rect), axis=0)
    # cv2.imshow('img', np.concatenate((disp1, disp2), axis=1))    
    # cv2.waitKey(0)
