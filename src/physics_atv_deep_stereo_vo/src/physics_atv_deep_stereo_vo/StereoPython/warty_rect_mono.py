import cv2
import numpy as np
from scipy.spatial.transform import Rotation
np.set_printoptions(suppress=True, precision=4)


# # the intrinsics from the camerainfo
# # parameters for the warty
# camleftk = np.array([ 1188.67771,     0.     ,   739.759  ,
#              0.     ,  1185.18512,   562.24167,
#              0.     ,     0.     ,     1.     ]).reshape(3,3)
# camleftd = np.array([-0.417066, 0.177679, -0.001099, -0.000515, 0.000000])
# R1 = np.array([ 1.,  0.,  0.,
#           0.,  1.,  0.,
#           0.,  0.,  1.]).reshape(3,3)
# P1 = np.array([  983.73999,     0.     ,   746.5113 ,     0.     ,
#              0.     ,  1073.05396,   565.99263,     0.     ,
#              0.     ,     0.     ,     1.     ,     0.     ]).reshape(3,4)        


# camrightk = np.array([ 1212.69789,     0.     ,   736.2487 ,
#              0.     ,  1208.40453,   556.26385,
#              0.     ,     0.     ,     1.     ]).reshape(3,3)
# camrightd = np.array([-0.415230, 0.175665, 0.000099, -0.000802, 0.000000])
# R2 = np.array([ 1.,  0.,  0.,
#           0.,  1.,  0.,
#           0.,  0.,  1.]).reshape(3,3)
# P2 = np.array( [ 1012.67322,     0.     ,   740.71055,     0.     ,
#              0.     ,  1099.31323,   559.81555,     0.     ,
#              0.     ,     0.     ,     1.     ,     0.     ]).reshape(3,4)


# parameters for the washy
camleftk = np.array([1174.332699776671, 0.0, 721.8011807658489, 0.0, 1171.470082927851, 573.4719662613194, 0.0, 0.0, 1.0]).reshape(3,3)
camleftd = np.array([-0.4312324432857076, 0.1995684500869378, -0.001496333995417086, 5.108433386006726e-05, 0.0])
R1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3)
P1 = np.array([964.4921875, 0.0, 722.1453890088014, 0.0, 0.0, 1053.176025390625, 579.0404877805558, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4)        


camrightk = np.array([1184.544846670511, 0.0, 713.0720819841532, 0.0, 1183.160419938363, 578.57816290504, 0.0, 0.0, 1.0]).reshape(3,3)
camrightd = np.array([-0.4342681600914292, 0.2125740039302204, -0.00178151263337098, 0.001392502079363557, 0.0])
R2 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3)
P2 = np.array( [979.6993408203125, 0.0, 713.6808465667273, 0.0, 0.0, 1066.025634765625, 584.7073738443141, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4)

imgsize = [1440, 1080]

map1, map2 = cv2.initUndistortRectifyMap(\
            camleftk, camleftd,\
            R1, P1,\
            imgsize, cv2.CV_32FC1)
leftremap = [map1, map2]

map3, map4 = cv2.initUndistortRectifyMap(\
            camrightk, camrightd,\
            R2, P2,\
            imgsize, cv2.CV_32FC1)
rightremap = [map3, map4]
# np.save('warty_stereo_maps.npy',[map1, map2, map3, map4])

# outleftdir = '/cairo/arl_bag_files/SARA/calibration/left'
# outrightdir = '/cairo/arl_bag_files/SARA/calibration/right'
# sourcedir = '/cairo/arl_bag_files/SARA/calibration/calibrationdata-pair'

outleftdir = '/cairo/arl_bag_files/SARA/calibration2/left'
outrightdir = '/cairo/arl_bag_files/SARA/calibration2/right'
sourcedir = '/cairo/arl_bag_files/SARA/calibration2/calibration_data'

for k in range(63):
    imgl = cv2.imread(sourcedir + '/left-'+str(k).zfill(4)+'.png')
    imgr = cv2.imread(sourcedir + '/right-'+str(k).zfill(4)+'.png')
    limg = cv2.remap( imgl, leftremap[0], leftremap[1], cv2.INTER_LINEAR )
    rimg = cv2.remap( imgr, rightremap[0], rightremap[1], cv2.INTER_LINEAR )
    cv2.imwrite(outleftdir+'/left-'+str(k).zfill(4)+'.png', limg)
    cv2.imwrite(outrightdir+'/right-'+str(k).zfill(4)+'.png', rimg)
    img = np.concatenate((limg, rimg), axis=1)
    for w in range(25, limg.shape[0], 50):
        img[w,:,:] = (0,255,255) 
    cv2.imwrite(outleftdir.split('/left')[0]+'/combine-'+str(k).zfill(4)+'.png', img)

# img = np.concatenate((limg, rimg), axis=1)
# for k in range(25, limg.shape[0], 50):
#     img[k,:,:] = (0,255,255) 
# cv2.imwrite('./combine_mono_rect.png', img)

import ipdb;ipdb.set_trace()
