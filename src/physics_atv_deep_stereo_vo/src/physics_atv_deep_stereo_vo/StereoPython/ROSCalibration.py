
from __future__ import print_function

import cv2
import numpy as np

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

def downsample_and_detect_corners(img, bSize, baseSize=307200):
    '''
    This function is implemented by referring to the ROS image_pipeline package.

    This function first downsample the input image to roughly the VGA size. 
    Then corners are detected on the downsampled image. After detection, there 
    is a refinement step on the downsampled image, followed by another refinement
    after the detected corner coordinates are scaled back to match the original
    image size.

    img (NumPy array, cv2 image): The input image, assumed to be a grayscale image.
    bSize (2-element): The chessboard size, cols x rows.
    baseSize (int): The pixel numbers of the downsample target. Default is 640x480.
    '''

    H, W = img.shape[:2]
    bCols, bRows = bSize[:2]

    scale = np.sqrt( H*W / ( 1.0 * baseSize ) )

    if ( scale > 1.0 ):
        imgScaled = cv2.resize( img, ( int( W/scale ), int( H/scale ) ) )
    else:
        imgScaled = img
    
    xScale = float( W ) / imgScaled.shape[1]
    yScale = float( H ) / imgScaled.shape[0]

    # Detect corners.
    detectionFlags = cv2.CALIB_CB_ADAPTIVE_THRESH \
        | cv2.CALIB_CB_NORMALIZE_IMAGE \
        | cv2.CALIB_CB_FAST_CHECK

    ok, corners = cv2.findChessboardCorners( 
        imgScaled, bSize, flags=detectionFlags)

    if not ok:
        return ok, corners

    # Check if the corners are within BORDER number of pixels to the image border. This should
    # already satisfied since the images may come from the camera_calibration of ROS. 
    BORDER = 8
    if not all([(BORDER < corners[i, 0, 0] < (W - BORDER)) and (BORDER < corners[i, 0, 1] < (H - BORDER)) for i in range(corners.shape[0])]):
        ok = False
        return ok, corners

    # Ensure that all corner-arrays are going from top to bottom.
    if bCols!=bRows:
        if corners[0, 0, 1] > corners[-1, 0, 1]:
            corners = np.copy(np.flipud(corners))
    else:
        directionCorners=( corners[-1]-corners[0]) >= np.array( [[0.0,0.0]] )

        if not np.all(directionCorners):
            if not np.any(directionCorners):
                corners = np.copy(np.flipud(corners))
            elif directionCorners[0][0]:
                corners=np.rot90(corners.reshape(bRows, bCols, 2)   ).reshape(bCols*bRows, 1, 2)
            else:
                corners=np.rot90(corners.reshape(bRows, bCols, 2), 3).reshape(bCols*bRows, 1, 2)

    # Refine the corners on the downsampled images.
    minDistDownsampled = float("inf")
    for row in range(bRows):
        for col in range(bCols - 1):
            index = row*bRows + col
            minDistDownsampled = min(minDistDownsampled, _pdist(corners[index, 0], corners[index + 1, 0]))

    for row in range(bRows - 1):
        for col in range(bCols):
            index = row*bRows + col
            minDistDownsampled = min(minDistDownsampled, _pdist(corners[index, 0], corners[index + bCols, 0]))

    radius = int(np.ceil(minDistDownsampled * 0.5))

    cv2.cornerSubPix( imgScaled, corners, (radius, radius), (-1,-1),
        ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ) )

    if ( scale > 1.0 ):
        # Re-scale the detected corners back to the original image size.
        cornersOriSize = corners.copy()
        cornersOriSize[:, :, 0] *= xScale
        cornersOriSize[:, :, 1] *= yScale

        radius = int( np.ceil( scale ) )
        cv2.cornerSubPix( img, cornersOriSize, ( radius, radius ), (-1, -1), 
            ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ) )
    else:
        cornersOriSize = corners

    return ok, cornersOriSize
