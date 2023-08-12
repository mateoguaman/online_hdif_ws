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
    disp = output_data.detach().cpu().squeeze().numpy() * 50 # 50 is the stereo normalization parameter
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
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path, save_engine=False):
    # initialize TensorRT engine and parse ONNX model
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
    if save_engine:
        with open('../models/'+onnx_file_path.split('/')[-1].split('.')[0]+'.engine', 'wb') as f:
            f.write(engine.serialize())
            print('Engine saved..')
    return engine, context

def load_engine(enginefile):
    with open(enginefile, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context    

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
    engine, context = build_engine(modelname)
    #enginename = curdir + '/../models/5_5_4_stereo_30000_sim.engine'
    #engine, context = load_engine(enginename)
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        # import ipdb;ipdb.set_trace()
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)

            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)

            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # preprocess input data
    starttime = time.time()
    thistime = starttime
    for k in range(10):

        host_input = np.array(preprocess_image("left_rect.png", "right_rect.png").numpy(), dtype=np.float32, order='C')
        cuda.memcpy_htod_async(device_input, host_input, stream)
        # run inference
        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()
        # postprocess results
        # import ipdb;ipdb.set_trace()
        output_data = torch.Tensor(host_output).reshape((output_shape[2],output_shape[3]))
        postprocess(output_data)

        print('===> TensorRT time {}'.format(time.time()-thistime))
        thistime = time.time()
    print('===> TensorRT total time {}'.format(time.time()-starttime))

if __name__ == '__main__':
    #pytorch_inference()
    main()
