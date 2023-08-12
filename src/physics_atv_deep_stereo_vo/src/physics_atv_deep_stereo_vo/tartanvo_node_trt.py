#!/usr/bin/env python3
from physics_atv_deep_stereo_vo.tartanvo_node import TartanSVONodeBase
from physics_atv_deep_stereo_vo.utils import Compose, CropCenter, Normalize, ResizeData

import rospy
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import physics_atv_deep_stereo_vo.common as common

import time

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class TartanSVONodeTensorRT(TartanSVONodeBase):

    def __init__(self):
        super(TartanSVONodeTensorRT, self).__init__()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = Compose([ResizeData((self.resize_h, self.resize_w)), 
                                  CropCenter((self.input_h, self.input_w)), 
                                  Normalize(mean=mean,std=std,keep_old=False)]) 
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor

    def load_model(self, ):
        # self.engine_file_path = self.curdir + '/../trt_models/43_6_2_vonet_30000_wo_pwc.trt'
        self.engine_file_path =  self.curdir + '/../trt_models/43_6_2_vonet_30000_wo_pwc_onnx2trt_16.trt'

    def transpose_np(self, sample): 
        kks = sample.keys()
        for kk in kks:
            data = sample[kk] 
            if len(data.shape) == 3: # transpose image-like data
                data = data.transpose(2,0,1)
                # add one dimention
                sample[kk] = data[None]
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)
                # add one dimention
                sample[kk] = data[None]

        return sample

    def main_loop(self,):
        '''
        Can not do TensorRT inference in the callback function
        The walk around is to buffer the data in the callback, and process it in the main loop
        '''
        r = rospy.Rate(self.loophz)

        f = open(self.engine_file_path, "rb")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context() 
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        rospy.loginfo("[VO node] TensorRT Engine Loaded...")

        while not rospy.is_shutdown(): # loop just for visualization
            datalen = len(self.imgbuf)
            if datalen>0:
                starttime = time.time()
                if datalen>1:
                    rospy.logwarn("[VO node] Buffer len {}, skip {} frames..".format(datalen, datalen-1))

                image_left_np, image_right_np, time_stamp = self.imgbuf[-1]
                self.imgbuf = []

                image_left_np = np.tile(image_left_np,(1,1,3))
                image_right_np = np.tile(image_right_np,(1,1,3))

                if self.last_left_img is not None:
                    sample = {'img0': self.last_left_img, 
                            'img0n': image_left_np, 
                            'img1': self.last_right_img}
                    sample = self.transform(sample)
                    sample = self.transpose_np(sample)
                    host_input = np.concatenate((sample['img0'],sample['img0n'],
                                                sample['img0'],sample['img1'],self.intrinsic_np), axis=1)
                    host_input = np.array(host_input, dtype=np.float32, order='C')
                    inputs[0].host = host_input
                    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                    motion = trt_outputs[0] * self.pose_std
                    # print(motion)
                    self.pub_motion(motion, time_stamp)

                self.last_left_img = image_left_np.copy()
                self.last_right_img = image_right_np.copy()
                rospy.loginfo("[VO node] vo inference time: {}:".format(time.time()-starttime))

            r.sleep()


if __name__ == '__main__':
    rospy.init_node("tartansvo_node", log_level=rospy.INFO)
    rospy.loginfo("tartanvo node initialized")
    node = TartanSVONodeTensorRT()
    node.main_loop()
