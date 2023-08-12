import torch

def loadPretrain(model, preTrainModel):
    preTrainDict = torch.load(preTrainModel)
    model_dict = model.state_dict()
    print('preTrainDict: {}'.format(preTrainDict.keys()))
    print('modelDict: {}'.format(model_dict.keys()))
    preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
    for item in preTrainDict:
        print('  Load pretrained layer: {}'.format(item))
    model_dict.update(preTrainDict)
    # for item in model_dict:
    # 	print '  Model layer: ',item
    model.load_state_dict(model_dict)
    return model

# from __future__ import division
# import torch
import math
import random
# from PIL import Image, ImageOps
import numpy as np
import numbers
import cv2

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor(object):
    def __call__(self, sample):
        kks = sample.keys()
        for kk in kks:
            data = sample[kk] 
            if len(data.shape) == 3: # transpose image-like data
                data = data.transpose(2,0,1)
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)
            data = data.astype(np.float32)
            sample[kk] = torch.from_numpy(data) # copy to make memory continuous

        return sample


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    This option should be before the to tensor
    """

    def __init__(self, mean, std, rgbbgr=False, keep_old=False):
        '''
        keep_old: keep both normalized and unnormalized data, 
        normalized data will be put under new key xxx_norm
        '''
        self.mean = mean
        self.std = std
        self.rgbbgr = rgbbgr
        self.keep_old = keep_old

    def __call__(self, sample):
        kks = list(sample.keys())
        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3 and img.shape[-1]==3: 
                img = img / 255.0
                if self.rgbbgr:
                    img = img[:,:,[2,1,0]] # bgr2rgb
                if self.mean is not None and self.std is not None:
                    for k in range(3):
                        img[:,:,k] = (img[:,:,k] - self.mean[k])/self.std[k]
                if self.keep_old:
                    sample[kk+'_norm'] = img
                    sample[kk] = sample[kk]/255.0
                else:
                    sample[kk] = img
        return sample

class CombineLR(object):
    '''
    combine the left and right image
    '''
    def __call__(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        rbgs = torch.cat((leftImg, rightImg),dim=0)
        return { 'rgbs':  rbgs, 'disp': sample['disp']}

class ResizeData(object):
    """Resize the data in a dict
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        kks = sample.keys()
        th, tw = self.size
        for kk in kks:
            if len(sample[kk].shape)==3:
                h, w = sample[kk].shape[0], sample[kk].shape[1]
                break
        if w == tw and h == th:
            return sample

        for kk in kks:
            if sample[kk] is None:
                continue
            if len(sample[kk].shape)==3 or len(sample[kk].shape)==2:
                sample[kk] = cv2.resize(sample[kk], (tw,th), interpolation=cv2.INTER_LINEAR)

        # change the intrinsics
        if 'blxfx' in kks:
            sample['blxfx'] = sample['blxfx']  * tw / w
        return sample

class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    if fix_ratio is False, w and h are resized separatedly
    if scale_w is given, w will be resized accordingly
    """

    def __init__(self, size, fix_ratio=True, scale_w=1.0, scale_disp=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fix_ratio = fix_ratio
        self.scale_w = scale_w
        self.scale_disp = scale_disp

    def __call__(self, sample):

        kks = sample.keys()
        th, tw = self.size
        for kk in kks:
            if len(sample[kk].shape)==3:
                hh, ww = sample[kk].shape[0], sample[kk].shape[1]
                break
        if ww == tw and hh == th:
            return sample
        # import ipdb;ipdb.set_trace()
        # resize the image if the image size is smaller than the target size
        scale_h, scale_w = 1., 1.
        if th > hh:
            scale_h = float(th)/hh
        if tw > ww:
            scale_w = float(tw)/ww
        if scale_h>1 or scale_w>1:
            if self.fix_ratio:
                scale_h = max(scale_h, scale_w)
                scale_w = max(scale_h, scale_w)
            w = int(round(ww * scale_w)) # w after resize
            h = int(round(hh * scale_h)) # h after resize
        else:
            w, h = ww, hh

        if self.scale_w != 1.0:
            scale_w = self.scale_w
            w = int(round(ww * scale_w))

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape)==3:
                if scale_h != 1. or scale_w != 1.:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw,:]
            elif len(img.shape)==2:
                if scale_h != 1. or scale_w != 1.:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw]

        if scale_h != 1. or scale_w != 1.: # adjust 
            # scale the flow
            scale_w = float(w)/ww
            scale_h = float(h)/hh
            if 'flow' in sample.keys():
                sample['flow'][:,:,0] = sample['flow'][:,:,0] * scale_w
                sample['flow'][:,:,1] = sample['flow'][:,:,1] * scale_h

            if self.scale_disp: # scale the depth
                if 'disp0' in sample.keys():
                    sample['disp0'][:,:] = sample['disp0'][:,:] * scale_w
                if 'disp1' in sample.keys():
                    sample['disp1'][:,:] = sample['disp1'][:,:] * scale_w
                if 'disp0n' in sample.keys():
                    sample['disp0n'][:,:] = sample['disp0n'][:,:] * scale_w
                if 'disp1n' in sample.keys():
                    sample['disp1n'][:,:] = sample['disp1n'][:,:] * scale_w
            else:
                sample['scale_w'] = np.array([scale_w ])# used in e2e-stereo-vo

        return sample

class RandomCrop(object):
    """Crops the given imgage(in numpy format) at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']

        (h, w, c) = leftImg.shape
        th, tw = self.size
        if w == tw and h == th:
            return {'left': leftImg, 'right': rightImg, 'disp': dispImg}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        leftImg = leftImg[y1:y1+th,x1:x1+tw,:]
        rightImg = rightImg[y1:y1+th,x1:x1+tw,:]
        dispImg = dispImg[y1:y1+th,x1:x1+tw,:]

        # print leftImg.shape,rightImg.shape,dispImg.shape

        return {'left': leftImg, 'right': rightImg, 'disp': dispImg}

class RandomHSV(object):
    """
    Change the image in HSV space
    """

    def __init__(self,HSVscale=(6,30,30)):
        self.Hscale, self.Sscale, self.Vscale = HSVscale


    def __call__(self, sample):
        leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']

        leftHSV = cv2.cvtColor(leftImg, cv2.COLOR_BGR2HSV)
        rightHSV = cv2.cvtColor(rightImg, cv2.COLOR_BGR2HSV)
        # change HSV
        h = random.random()*2-1
        s = random.random()*2-1
        v = random.random()*2-1
        leftHSV[:,:,0] = np.clip(leftHSV[:,:,0]+self.Hscale*h,0,255)
        leftHSV[:,:,1] = np.clip(leftHSV[:,:,1]+self.Sscale*s,0,255)
        leftHSV[:,:,2] = np.clip(leftHSV[:,:,2]+self.Vscale*v,0,255)
        rightHSV[:,:,0] = np.clip(rightHSV[:,:,0]+self.Hscale*h,0,255)
        rightHSV[:,:,1] = np.clip(rightHSV[:,:,1]+self.Sscale*s,0,255)
        rightHSV[:,:,2] = np.clip(rightHSV[:,:,2]+self.Vscale*v,0,255)

        leftImg = cv2.cvtColor(leftHSV,cv2.COLOR_HSV2BGR)
        rightImg = cv2.cvtColor(rightHSV,cv2.COLOR_HSV2BGR)

        return {'left': leftImg, 'right': rightImg, 'disp': dispImg}

def combine2img(sample,mean,std):
    """
    convert dict of tensor to dict of numpy array, for visualization
    """
    rgbs = sample['rgbs']
    dispImg = sample['disp']
    if len(rgbs.size())==4:
        rgbs = rgbs[0]
        dispImg = dispImg[0,0]
    leftImg = rgbs[0:3,:,:]
    rightImg = rgbs[3:,:,:]
    
    return sample2img({'left':leftImg, 'right': rightImg, 'disp':dispImg}, mean, std)

class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample): 
        if self.downscale==1:
            return sample

        if 'flow' in sample:
            sample['flow'] = cv2.resize(sample['flow'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if 'intrinsic' in sample:
            sample['intrinsic'] = cv2.resize(sample['intrinsic'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if 'fmask' in sample:
            sample['fmask'] = cv2.resize(sample['fmask'],
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
            
        if 'disp0' in sample:
            sample['disp0'] = cv2.resize(sample['disp0'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        return sample


def sample2img(sample,mean,std):
    """
    convert dict of tensor to dict of numpy array, for visualization
    """
    leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']
    if len(leftImg.size())==4:
        leftImg = leftImg[0,:,:,:]
        rightImg = rightImg[0,:,:,:]
        dispImg = dispImg[0,0,:,:]
    leftImg = tensor2img(leftImg, mean, std)
    rightImg = tensor2img(rightImg,mean,std)

    dispImg = dispImg.numpy() #.transpose(1,2,0)

    return {'left':leftImg, 'right': rightImg, 'disp':dispImg}

def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)*float(255)).astype(np.uint8)
    return tensImg


def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer

from scipy.spatial.transform import Rotation as R

def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix() #as_dcm()

def se2SE(se_data):
    '''
    6 -> 4 x 4
    '''
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat

def SO2quat(SO_data):
    rr = R.from_matrix(SO_data) #from_dcm(SO_data)
    return rr.as_quat()

def se2quat(se_data):
    '''
    6 -> 7
    '''
    SE_mat = se2SE(se_data)
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_mat[0:3,0:3])
    pos_quat[:3] = SE_mat[0:3,3].T
    return pos_quat

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr

def visdepth(depth, maxthresh = 50):
    depthvis = np.clip(depth,0,maxthresh)
    depthvis = depthvis/maxthresh*255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))

    return depthvis

def disp2vis(disp, scale=10,):
    '''
    disp: h x w float32 numpy array
    return h x w x 3 uint8 numpy array
    '''
    disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    return disp_color


class ColorCorrection(object):
    """
    """
    def init_gamma_curves(self):
        gamma_curves = []
        gammas = np.arange(0.1,1.01,0.005)
        for gamma in gammas:
            gamma_curve = np.empty((256), np.uint8)
            for i in range(256):
                gamma_curve[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            gamma_curves.append(gamma_curve)
        return np.array(gamma_curves), gammas

    def __init__(self):
        self.gamma_curves, self.gammas = self.init_gamma_curves()
        self.keylist = ['img0', 'img0n', 'img1', 'left', 'right']

    def autowhitebalance(self, img_original, img_additional=None):
        img_small = cv2.resize(img_original, (0,0), fx=0.2, fy=0.2) # resize the image to make it faster
        # whitepatch = np.percentile(img_small, 95, axis=(0,1))
        meanvalue = np.mean(img_small, axis=(0,1))
        # print(whitepatch, meanvalue)

        res = np.clip(img_original*128.0/meanvalue.reshape(1,1,3), 0, 255).astype(np.uint8)
        # img_percentile = np.clip(img_small*255.0/whitepatch.reshape(1,1,3), 0, 255).astype(np.uint8)
        if img_additional is not None: # used in the stereo case
            res2 = np.clip(img_additional*128.0/meanvalue.reshape(1,1,3), 0, 255).astype(np.uint8)
            res = (res, res2)

        return res

    def get_hist_curve(self, img):
        img_small = cv2.resize(img, (0,0), fx=0.2, fy=0.2) # resize the image to make it faster
        hist, _ = np.histogram(img_small[:], range=(0,256), bins=256)
        hist_acc = np.cumsum(hist)
        hist_acc = hist_acc / hist_acc[-1]
        lookUpTable = np.clip(hist_acc * 255.0, 0, 255).astype(np.uint8)

        return lookUpTable


    def find_gamma_curve(self, img, gamma_curves, gammas, soft_factor = 1.0):
        img_curve = self.get_hist_curve(img)
        # find closest match
        diff = gamma_curves.astype(np.float32) - img_curve.reshape(1,256).astype(np.float32)
        diff = np.abs(diff).mean(axis=1)
        min_ind = np.argmin(diff)

        # soft the gamma curve further
        gamma = 1-(1-gammas[min_ind]) * soft_factor
        soft_ind = np.argmin(np.abs(gammas - gamma))
        # print("closese gamma {}, soft gamma {}".format(gammas[min_ind], gammas[soft_ind]))

        return gamma_curves[soft_ind]

    def gentleGammaCorrection(self, img_original, gamma_curves, gammas, img_additional=None, soft_factor = 1.0):
        gamma_curve = self.find_gamma_curve(img_original, gamma_curves, gammas, soft_factor)
        res = cv2.LUT(img_original, gamma_curve)
        if img_additional is not None: # used in the stereo case
            res2 = cv2.LUT(img_additional, gamma_curve)
            res = (res, res2)
        return res  

    def correct_image(self, img):
        img_correction = self.gentleGammaCorrection(img, self.gamma_curves, self.gammas, soft_factor=0.7)
        img_correction_awb = self.autowhitebalance(img_correction)
        return img_correction_awb

    def __call__(self, sample):
        for key in self.keylist:
            if key in sample:
                sample[key] = self.correct_image(sample[key])

        return sample
