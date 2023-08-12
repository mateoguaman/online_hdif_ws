import cv2
import torch
import numpy as np
import time
import os

from physics_atv_deep_stereo_vo.StereoVONet import StereoVONet

# after convert the model
# run onnxsim to simplify the model
# python3 -m onnxsim 43_6_2_vonet_30000_wo_pwc.onnx 43_6_2_vonet_30000_wo_pwc_sim.onnx
# https://github.com/daquexian/onnx-simplifier

# [TensorRT] ERROR: INVALID_ARGUMENT: getPluginCreator could not find plugin ScatterND version 1
# ===> Installed tensorrt 8.0 using jetson 4.6


# Reading engine from file 43_6_2_vonet_30000_wo_pwc_sim.trt
#[TensorRT] ERROR: 3: getPluginCreator could not find plugin: ScatterND version: 1
#[TensorRT] ERROR: 1: [pluginV2Runner.cpp::load::292] Error Code 1: Serialization (Serialization assertion creator failed.Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)
#[TensorRT] ERROR: 4: [runtime.cpp::deserializeCudaEngine::76] Error Code 4: Internal Error (Engine deserialization failed.)
# ===> add one line in the code: trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# when building the engine: 
# [TensorRT] ERROR: Tactic Device request: 2100MB Available: 1536MB. Device memory is insufficient to use tactic.
# [TensorRT] WARNING: Skipping tactic 3 due to oom error on requested size of 2100 detected for tactic 4.
# the inference time only went from 0.25 to 0.2 (not sure if it's because of those building errors)
# ===> download and build https://github.com/onnx/onnx-tensorrt
# ===> cd /home/xavier1/workspace/onnx-tensorrt/build
# ===> ./onnx2trt /home/xavier1/ros_atv/src/physics_atv_deep_stereo_vo/models/43_6_2_vonet_30000_wo_pwc.onnx -o 43_6_2_vonet_30000_wo_pwc_onnx2trt.trt -d 16 -b 1
# ===> the speed is similar using onnx2trt

def scale_imgs( sample, input_w, input_h):
    leftImg, rightImg, leftImg2 = sample['left'], sample['right'], sample['left2']
    leftImg = cv2.resize(leftImg,(input_w, input_h))
    rightImg = cv2.resize(rightImg,(input_w, input_h))
    leftImg2 = cv2.resize(leftImg2,(input_w, input_h))

    return {'left': leftImg, 'right': rightImg, 'left2': leftImg2}

def crop_imgs( sample, crop_w, crop_h):
    leftImg, rightImg, leftImg2 = sample['left'], sample['right'], sample['left2']
    if crop_w>0:
        leftImg = leftImg[:, crop_w:-crop_w, :]
        rightImg = rightImg[:, crop_w:-crop_w, :]
        leftImg2 = leftImg2[:, crop_w:-crop_w, :]
    if crop_h>0:
        leftImg = leftImg[crop_h:-crop_h, :, :]
        rightImg = rightImg[crop_h:-crop_h, :, :]
        leftImg2 = leftImg2[crop_h:-crop_h, :, :]
    return {'left': leftImg, 'right': rightImg, 'left2': leftImg2}

def to_tensor(sample):
    leftImg, rightImg, leftImg2 = sample['left'], sample['right'], sample['left2']
    leftImg = leftImg.transpose(2,0,1)/float(255)
    rightImg = rightImg.transpose(2,0,1)/float(255)
    leftImg2 = leftImg2.transpose(2,0,1)/float(255)
    leftTensor = torch.from_numpy(leftImg).float()
    rightTensor = torch.from_numpy(rightImg).float()
    leftTensor2 = torch.from_numpy(leftImg2).float()
    # rgbsTensor = torch.cat((leftTensor, rightTensor), dim=0)
    return {'left': leftTensor, 'right': rightTensor, 'left2': leftTensor2}

def normalize(sample, mean, std):
    leftImg, rightImg, leftImg2 = sample['left'], sample['right'], sample['left2']
    leftImg_norm = leftImg.clone()
    rightImg_norm = rightImg.clone()
    leftImg2_norm = leftImg2.clone()
    for t, m, s in zip(leftImg_norm, mean, std):
        t.sub_(m).div_(s)
    for t, m, s in zip(rightImg_norm, mean, std):
        t.sub_(m).div_(s)
    for t, m, s in zip(leftImg2_norm, mean, std):
        t.sub_(m).div_(s)
    return {'left': leftImg, 'left2': leftImg2, 
            'left_norm': leftImg_norm, 'right_norm': rightImg_norm, 'left2_norm': leftImg2_norm}

def make_intrinsics_layer(w = 1024, h = 544, fx = 477.6049499511719, fy = 477.6049499511719, ox = 499.5, oy = 252.0):

    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    # handle resize - hard code
    intrinsicLayer = cv2.resize(intrinsicLayer,(844, 448))
    # handle crop 
    intrinsicLayer = intrinsicLayer[:, 102:-102, :]
    # handle downsampleflow
    # intrinsicLayer = cv2.resize(intrinsicLayer,(160, 112))    
    # handle to_tensor
    intrinsicLayer = intrinsicLayer.transpose(2, 0, 1)
    intrinsicLayer = torch.from_numpy(intrinsicLayer).float()
    intrinsicLayer = intrinsicLayer.unsqueeze(0)

    return intrinsicLayer

def preprocess_image(leftimg_path, rightimg_path, leftimg2_path, intrinsicLayer):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    # read input image
    left_img = cv2.imread(leftimg_path)
    right_img = cv2.imread(rightimg_path)
    left2_img = cv2.imread(leftimg2_path)
    sample = {'left': left_img, 'right': right_img, 'left2': left2_img}
    sample = scale_imgs(sample, 844, 448)
    sample = crop_imgs(sample, 102, 0)
    sample = to_tensor(sample)
    sample = normalize(sample, mean, std)
    img0_flow = sample['left_norm'].unsqueeze(0)
    img1_flow = sample['left2_norm'].unsqueeze(0)
    img0_stereo = sample['left_norm'].unsqueeze(0)
    img1_stereo = sample['right_norm'].unsqueeze(0)
    inputTensor = torch.cat((img0_flow, img1_flow, img0_stereo, img1_stereo, intrinsicLayer), dim=1)

    return inputTensor

def disp2vis(disp, scale=10,):
    '''
    disp: h x w float32 numpy array
    return h x w x 3 uint8 numpy array
    '''
    disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
    disp = np.tile(disp[:,:,np.newaxis], (1, 1, 3))
    # disp = cv2.resize(disp,(640,480))

    return disp

def postprocess(output_data):
    output_np = output_data.detach().cpu().squeeze().numpy() 
    print(output_np)
 
def load_model(model, modelname):
    preTrainDict = torch.load(modelname)
    model_dict = model.state_dict()
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

    model_dict.update(preTrainDict)
    model.load_state_dict(model_dict)
    return model


def convert_model():
    curdir = os.path.dirname(os.path.realpath(__file__))
    modelname = curdir + '/../models/43_6_2_vonet_30000_wo_pwc.pkl'
    vonet = StereoVONet(network=2, intrinsic=True, 
                        flowNormFactor=1.0, stereoNormFactor=0.02, poseDepthNormFactor=0.25, 
                        down_scale=True, config=1, 
                        fixflow=True, fixstereo=True, autoDistTarget=False,
                        blxfx=100.14994812011719*844/1024) # consider the scale width 844
    load_model(vonet, modelname)
    print('deep model loaded..')
    vonet.cuda()
    vonet.eval()

    intrinsicLayer = make_intrinsics_layer()

    # inputTensor = preprocess_image("left_rect.png", "right_rect.png", "left_rect2.png", intrinsicLayer)
    inputTensor = preprocess_image("left.png", "right.png", "left2.png", intrinsicLayer)
    output = vonet(inputTensor.cuda())
    postprocess(output)

    ONNX_FILE_PATH = '43_6_2_vonet_30000_wo_pwc.onnx'
    torch.onnx.export(vonet, inputTensor.cuda(), ONNX_FILE_PATH, input_names=['input'],
                      output_names=['output'], export_params=True) #, opset_version=11

    print('convert success')

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import physics_atv_deep_stereo_vo.common as common
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    import ipdb;ipdb.set_trace()
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 8GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 33
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    #if builder.platform_has_fast_fp16:
    builder.fp16_mode = False

    # last_layer = network.get_layer(network.num_layers - 1)
    # network.mark_output(last_layer.get_output(0))

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 34 # 4GB
            config.set_flag(trt.BuilderFlag.FP16)
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 14, 448, 640]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def pytorch_inference():
    curdir = os.path.dirname(os.path.realpath(__file__))
    modelname = curdir + '/../models/43_6_2_vonet_30000_wo_pwc.pkl'
    vonet = StereoVONet(network=2, intrinsic=True, 
                        flowNormFactor=1.0, stereoNormFactor=0.02, poseDepthNormFactor=0.25, 
                        down_scale=True, config=1, 
                        fixflow=True, fixstereo=True, autoDistTarget=False,
                        blxfx=100.14994812011719*844/1024) # consider the scale width 844
    load_model(vonet, modelname)
    print('deep model loaded..')
    vonet.cuda()
    vonet.eval()

    intrinsicLayer = make_intrinsics_layer()

    starttime = time.time()
    thistime = starttime
    for k in range(10):
        inputTensor = preprocess_image("left.png", "right.png", "left2.png", intrinsicLayer)
        output = vonet(inputTensor.cuda())
        postprocess(output)

        print('===> Pytorch time {}'.format(time.time()-thistime))
        thistime = time.time()
    print('===> Pytorch total time {}'.format(time.time()-starttime))


def main():
    # initialize TensorRT engine and parse ONNX model
    curdir = os.path.dirname(os.path.realpath(__file__))
    modelname = curdir + '/../trt_models/43_6_2_vonet_30000_wo_pwc.onnx'
    engine_file_path = curdir + '/../trt_models/43_6_2_vonet_30000_wo_pwc_onnx2trt_16.trt'

    with get_engine(modelname, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image ...')
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        intrinsicLayer = make_intrinsics_layer()
        starttime = time.time()
        thistime = starttime
        for k in range(50):
            # import ipdb;ipdb.set_trace()
            host_input = np.array(preprocess_image("left.png", "right.png", "left2.png", intrinsicLayer).numpy(), dtype=np.float32, order='C')
            inputs[0].host = host_input
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(trt_outputs[0])
            print('===> TensorRT time {}'.format(time.time()-thistime))
            thistime = time.time()
        print('===> TensorRT total time {}'.format(time.time()-starttime))

    print('engine loaded...')

if __name__ == '__main__':
    #convert_model()
    #pytorch_inference()
    main()
