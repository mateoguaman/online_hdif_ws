#!/usr/bin/env python3

import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

from std_msgs.msg import Header

import time
# from PIL import Image


class DummyNode:
    def __init__(self):
        self.loophz = 100
        self.databuf = []

        self.bridge = CvBridge()

        self.dummy_pub = rospy.Publisher('dummy_img', Image, queue_size=1)

        leftimg_sync = message_filters.Subscriber('/multisense/left/image_rect', Image)
        rightimg_sync = message_filters.Subscriber('/multisense/right/image_rect', Image)
        colorimg_sync = message_filters.Subscriber('/multisense/left/image_rect_color', Image)
        ts = message_filters.TimeSynchronizer([leftimg_sync, rightimg_sync, colorimg_sync], 10)
        ts.registerCallback(self.handle_imgs)

    def handle_imgs(self, leftmsg, rightmsg, colormsg):
        left_image_np = np.frombuffer(leftmsg.data, dtype=np.uint8).reshape(leftmsg.height, leftmsg.width, -1)
        right_image_np = np.frombuffer(rightmsg.data, dtype=np.uint8).reshape(rightmsg.height, rightmsg.width, -1)
        color_image_np = np.frombuffer(colormsg.data, dtype=np.uint8).reshape(colormsg.height, colormsg.width, -1)

        self.databuf.append([left_image_np, right_image_np, color_image_np, leftmsg.header.stamp])

    def pub_dummy(self, dummy_img, timestamp):
        header = Header()

        # dummy_img = Image.fromarray(dummy_img.astype(np.uint8))
        img_msg = self.bridge.cv2_to_imgmsg(dummy_img, "rgb8")
        img_msg.header.stamp = timestamp

        self.dummy_pub.publish(img_msg)

    def get_silly_img(self, color_img_np):

        color_img_np = cv2.cvtColor(color_img_np, cv2.COLOR_BGR2RGB)

        sillyT = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            torchvision.transforms.RandomPosterize(bits=3,p=1)
        ])

        silly_img = sillyT(color_img_np)
        silly_img = silly_img.permute(1,2,0).cpu().numpy()

        return silly_img

    def main_loop(self):
        r = rospy.Rate(self.loophz)

        while not rospy.is_shutdown():
            datalen = len(self.databuf)
            if datalen>0:
                starttime = time.time()
                if datalen>1:
                    rospy.logwarn("[Stereo node] Buffer len {}, skip {} frames..".format(datalen, datalen-1))
                left_image_np, right_image_np, color_image_np, time_stamp = self.databuf[-1]
                self.databuf = []

                silly_img = self.get_silly_img(color_image_np)

                self.pub_dummy(silly_img, time_stamp)



if __name__ == '__main__':

    rospy.init_node("dummy_node", log_level=rospy.INFO)
    rospy.loginfo("dummy_node initialized")

    node = DummyNode()
    node.main_loop()

