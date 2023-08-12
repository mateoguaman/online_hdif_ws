#!/usr/bin/env python3
import rospy
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import physics_atv_deep_stereo_vo.common as common

import time

from physics_atv_deep_stereo_vo.stereo_node_multisense import StereoNodeBase

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class StereoNodeTensorRT(StereoNodeBase):

    def __init__(self):
        super(StereoNodeTensorRT, self).__init__()

    def load_model(self, ):
        self.engine_file_path = self.curdir + '/../trt_models/5_5_4_stereo_30000_sim_onnx2trt_2G_1B.trt'
        # self.engine_file_path =  self.curdir + '/../trt_models/5_5_4_stereo_30000_sim_onnx2trt_16.trt'


    def normalize(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        leftImg = leftImg.transpose(2,0,1)/float(255)
        rightImg = rightImg.transpose(2,0,1)/float(255)
        for k in range(3):
            leftImg[k,:,:] = (leftImg[k,:,:]-self.mean[k])/self.std[k]
            rightImg[k,:,:] = (rightImg[k,:,:]-self.mean[k])/self.std[k]
        return {'left': leftImg, 'right': rightImg}

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
        rospy.loginfo("[Stereo node] TensorRT Engine Loaded...")

        while not rospy.is_shutdown(): # loop just for visualization
            datalen = len(self.databuf)
            if datalen>0:
                starttime = time.time()
                if datalen>1:
                    rospy.logwarn("[Stereo node] Buffer len {}, skip {} frames..".format(datalen, datalen-1))
                left_image_np, right_image_np, color_image_np, time_stamp = self.databuf[-1]
                self.databuf = []
                sample, color_image_np = self.imgs_processing(left_image_np, right_image_np, color_image_np)
                sample = self.normalize(sample) 
                host_input = np.concatenate((np.expand_dims(sample['left'], axis=0),np.expand_dims(sample['right'], axis=0)), axis=1)
                host_input = np.array(host_input, dtype=np.float32, order='C')
                inputs[0].host = host_input
                # import ipdb;ipdb.set_trace()
                trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                disp = trt_outputs[0].reshape((self.input_h, self.input_w)) * 50

                self.pub_pc2(disp, time_stamp, color_image_np)
                rospy.loginfo_throttle(1.0, '[Stereo node] Inference time {}'.format(time.time()-starttime))
                if self.visualize:
                    self.visualize_depth(disp)
            r.sleep()

if __name__ == '__main__':

    rospy.init_node("stereo_net", log_level=rospy.INFO)
    rospy.loginfo("stereo_net_node initialized")

    node = StereoNodeTensorRT()
    node.main_loop()                