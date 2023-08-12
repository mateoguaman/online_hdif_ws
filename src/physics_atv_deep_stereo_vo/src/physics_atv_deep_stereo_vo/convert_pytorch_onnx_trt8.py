import cv2
import torch
import numpy as np
import time
import os

from physics_atv_deep_stereo_vo.StereoNet7 import StereoNet7 as StereoNet


def scale_imgs( sample, input_w, input_h):
    leftImg, rightImg = sample['left'], sample['right']
    leftImg = cv2.resize(leftImg,(input_w, input_h))
    rightImg = cv2.resize(rightImg,(input_w, input_h))

    return {'left': leftImg, 'right': rightImg}

def crop_imgs( sample, crop_w, crop_h):
    leftImg, rightImg = sample['left'], sample['right']
    leftImg = leftImg[crop_h:-crop_h, crop_w:-crop_w, :]
    rightImg = rightImg[crop_h:-crop_h, crop_w:-crop_w, :]

    return {'left': leftImg, 'right': rightImg}

def to_tensor(sample):
    leftImg, rightImg = sample['left'], sample['right']
    leftImg = leftImg.transpose(2,0,1)/float(255)
    rightImg = rightImg.transpose(2,0,1)/float(255)
    leftTensor = torch.from_numpy(leftImg).float()
    rightTensor = torch.from_numpy(rightImg).float()
    # rgbsTensor = torch.cat((leftTensor, rightTensor), dim=0)
    return {'left': leftTensor, 'right': rightTensor}


def normalize(sample, mean, std):
    leftImg, rightImg = sample['left'], sample['right']
    for t, m, s in zip(leftImg, mean, std):
        t.sub_(m).div_(s)
    for t, m, s in zip(rightImg, mean, std):
        t.sub_(m).div_(s)
    return {'left': leftImg, 'right': rightImg}

def preprocess_image(leftimg_path, rightimg_path):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
     
    # read input image
    left_img = cv2.imread(leftimg_path)
    right_img = cv2.imread(rightimg_path)
    sample = {'left': left_img, 'right': right_img}
    sample = crop_imgs(sample, 64, 32)
    sample = scale_imgs(sample, 512, 256)
    sample = to_tensor(sample)
    sample = normalize(sample, mean, std)
    leftTensor = sample['left'].unsqueeze(0)
    rightTensor = sample['right'].unsqueeze(0)
    inputTensor = torch.cat((leftTensor, rightTensor), dim=1)

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
    disp = output_data * 50 # 50 is the stereo normalization parameter
    dispvis = disp2vis(disp, scale=3)

    cv2.imwrite('output.png', dispvis)
 
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
    modelname = curdir + '/../models/5_5_4_stereo_30000.pkl'
    stereonet = StereoNet(group_norm=False)
    load_model(stereonet, modelname)
    print('deep model loaded..')
    stereonet.cuda()
    stereonet.eval()

    inputTensor = preprocess_image("left_rect.png", "right_rect.png")
    output = stereonet(inputTensor.cuda())
    postprocess(output)

    ONNX_FILE_PATH = '5_5_4_stereo_30000.onnx'
    torch.onnx.export(stereonet, inputTensor, ONNX_FILE_PATH, input_names=['input'],
                      output_names=['output'], export_params=True, opset_version=11)

    print('convert success')

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import physics_atv_deep_stereo_vo.common as common
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

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
            network.get_input(0).shape = [1, 6, 256, 512]
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
    modelname = curdir + '/../models/5_5_4_stereo_30000.pkl'
    stereonet = StereoNet(group_norm=False)
    load_model(stereonet, modelname)
    print('deep model loaded..')
    stereonet.cuda()
    stereonet.eval()

    starttime = time.time()
    thistime = starttime
    for k in range(10):
        inputTensor = preprocess_image("left_rect.png", "right_rect.png")
        output = stereonet(inputTensor.cuda())
        postprocess(output)

        print('===> Pytorch time {}'.format(time.time()-thistime))
        thistime = time.time()
    print('===> Pytorch total time {}'.format(time.time()-starttime))



def main():
    # initialize TensorRT engine and parse ONNX model
    curdir = os.path.dirname(os.path.realpath(__file__))
    modelname = curdir + '/../trt_models/5_5_4_stereo_30000.onnx'
    engine_file_path = '5_5_4_stereo_30000_sim_onnx2trt_2G_1B.trt'

    with get_engine(modelname, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image ...')
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.

        # preprocess input data
        starttime = time.time()
        thistime = starttime
        for k in range(10):

            host_input = np.array(preprocess_image("left_rect.png", "right_rect.png").numpy(), dtype=np.float32, order='C')
            inputs[0].host = host_input
            import ipdb;ipdb.set_trace()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            output_data = trt_outputs[0].reshape((256,512))
            postprocess(output_data)

            print('===> TensorRT time {}'.format(time.time()-thistime))
            thistime = time.time()
        print('===> TensorRT total time {}'.format(time.time()-starttime))

if __name__ == '__main__':
    #pytorch_inference()
    main()
