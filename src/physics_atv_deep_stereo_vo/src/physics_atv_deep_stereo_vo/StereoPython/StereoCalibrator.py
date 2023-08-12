
from __future__ import print_function

import cv2
import numpy as np
import glob
import copy
import json
import os
import math

from ROSCalibration import downsample_and_detect_corners

G_IDX_L = 0
G_IDX_R = 1

# FN_IMAGE_SIZE            = "ImageSize.txt"
FN_IMAGE_SIZE            = "ImageSize.json"
FN_CAMERA_MATRIX_LEFT    = "CameraMatrixLeft.dat"
FN_CAMERA_MATRIX_RIGHT   = "CameraMatrixRight.dat"
FN_DISTORTION_COEFFICIENT_LEFT  = "DistortionCoefficientLeft.dat"
FN_DISTORTION_COEFFICIENT_RIGHT = "DistortionCoefficientRight.dat"
FN_ESSENTIAL_MATRIX      = "E.dat"
FN_FUNDAMENTAL_MATRIX    = "F.dat"
FN_ROTATION_MATRIX       = "R.dat"
FN_TRANSLATION_MATRIX    = "T.dat"
FN_IMAGE_FILE_LIST_LEFT  = "ImageFileList_Left.txt"
FN_IMAGE_FILE_LIST_RIGHT = "ImageFileList_Right.txt"
FN_OBJECT_INDEX          = "objIdx.dat"
FN_R1                    = "R1.dat"
FN_R2                    = "R2.dat"
FN_P1                    = "P1.dat"
FN_P2                    = "P2.dat"
FN_Q                     = "Q.dat"
FN_ROI_1                 = "ROI_1.dat"
FN_ROI_2                 = "ROI_2.dat"
FN_CB_GRID_SIZE          = "CBGridSize.dat"
FN_CB_SQUARE_SIZE        = "CBSquareSize.dat"

DR_IMAGE_POINTS_LEFT     = "ImagePointsLeft"
DR_IMAGE_POINTS_RIGHT    = "ImagePointsRight"
DR_RECTIFIED_IMAGES      = "Rectified"

PF_IMAGE_POINTS_LEFT     = "Left"
PF_IMAGE_POINTS_RIGHT    = "Right"

REMAP_CANVAS_LENGTH      = 1280

RECTIFIED_RECT_THICKNESS = 3
RECTIFIED_LINE_THICKNESS = 1
RECTIFIED_RECT_COLOR     = (0,   0, 255)
RECTIFIED_LINE_COLOR_0   = (0, 255,   0)
RECTIFIED_LINE_COLOR_1   = (0, 155, 255)
RECTIFIED_LINE_COLOR_LIST = [ RECTIFIED_LINE_COLOR_0, RECTIFIED_LINE_COLOR_1 ]

def print_separator(c, n = 50, title = None):
    """
    Print a visual separator with c as the repetitive character.
    c - The character.
    n - The number of characters that forming the separator on the screen.
    """

    # Create a list.
    if ( title is not None ):
        nTitle = len( title ) + 2

        if ( nTitle < n - 1 ):
            nHalfSep = int( ( n - nTitle ) / 2 )
        else:
            nHalfSep = 1
        
        sep = c * nHalfSep + " " + title + " " + c * nHalfSep
    else:
        sep = c * n

    # Print.
    print(sep)

def i_r(v):
    return int(round(v))

class StereoCalibrator(object):
    def __init__(self, cbSize, cbSquareSize, 
        imagePathLeft, imagePathRight, outputPath,
        downsample):
        self.cbSize         = cbSize # Column, row order.
        self.cbSquareSize   = cbSquareSize
        self.imagePathLeft  = imagePathLeft
        self.imagePathRight = imagePathRight
        self.outputPath     = outputPath
        self.imagePointsOutDir = [ self.outputPath + '/' + DR_IMAGE_POINTS_LEFT,\
            self.outputPath + '/' + DR_IMAGE_POINTS_RIGHT ]

        self.imageFileList  = [] # 2 dimension list.
        self.nImages        = 0

        self.goodImages  = [] # 2 dimension list.
        self.nGoodImages = 0

        self.objectPoints = []
        self.imagePoints  = [] # 2 dimension list.

        self.imageSize   = 0
        self.imageHeight = 0
        self.imageWidth  = 0
        self.downsample  = downsample

        self.cameraMatrix = []
        self.distortionCoefficients = []
        self.isIndividuallyCalibrated = False
        self.R = None
        self.T = None
        self.E = None
        self.F = None

        self.isCalibrated = False

        self.R1   = None
        self.R2   = None
        self.P1   = None
        self.P2   = None
        self.Q    = None
        self.Roi1 = None
        self.Roi2 = None

        self.isRectified = False

        self.remapCanvasLength = REMAP_CANVAS_LENGTH

        self.filenamePattern = "*.jpg"

        self.silence = False

        self.isDebugging = False
        self.debuggingNImagePairs = 3
        self.WARN_LIMIT_IMAGE_POINTS = 9999

    def showInfo(self, infoString, newline = True):
        if ( False == self.silence ):
            if ( True == newline ):
                print(infoString)
            else:
                print(infoString, end = '')

    def dump_to_file(self, obj, fn):
        """Dump the content of obj to fn."""

        try:
            with open(fn, 'w') as f:
                json.dump(obj, f, separators = (', \n', ':'))
        except IOError as e:
            print("Couldn't open file with exception (%s)." % e)

    def prepare_file_names(self):
        """Prepare the filenames and check if the numbers of the files for each camera is the same."""

        # Clear data.
        self.imageFileList = []

        self.imageFileList.append( sorted(glob.glob( self.imagePathLeft  + "/" + self.filenamePattern )) )
        self.imageFileList.append( sorted(glob.glob( self.imagePathRight + "/" + self.filenamePattern )) )

        # Dump the image file lists to files.
        self.dump_to_file(self.imageFileList[G_IDX_L], self.outputPath + '/' + FN_IMAGE_FILE_LIST_LEFT)
        self.dump_to_file(self.imageFileList[G_IDX_R], self.outputPath + '/' + FN_IMAGE_FILE_LIST_RIGHT)

        nImagesLeft  = len( self.imageFileList[G_IDX_L] )
        nImagesRight = len( self.imageFileList[G_IDX_R] )

        if ( nImagesLeft != nImagesRight ):
            raise Exception("Error: The numbers of images for the cameras are not consistent. nImagesLeft = %d, nImageRight = %d." % (nImagesLeft, nImagesRight))
            
        if ( nImagesLeft == 0 ):
            raise Exception("Error: 0 images found at %s or %s with pattern \"%s\"" % \
                ( self.imagePathLeft, self.imagePathRight, self.filenamePattern ))
        
        self.nImages = nImagesLeft

        if ( True == self.isDebugging ):
            self.nImages = self.debuggingNImagePairs

        return True

    def write_image_points_single_camera(self, folderName, ips, prefix = None):
        """
        folderName - The folder name.
        ips - The image points list which as a 4D structure.
        """

        # Get the size of the list.
        nIPS = len( ips )
        if ( nIPS > self.WARN_LIMIT_IMAGE_POINTS ):
            print("The number of image points (%d) is more than the limit (%d)." % (nIPS, WARN_LIMIT_IMAGE_POINTS))

        shape = ips[0].shape

        # The counter.
        I = 0

        for ip in ips:
            # Compose the file name.
            if ( prefix is None ):
                fn = "%04d.dat" % I
            else:
                fn = "%s_%04d.dat" % ( prefix, I )

            fn = folderName + '/' + fn

            # Resize the image points.
            ipR = ip.reshape( (shape[0], shape[2]) )

            # Write the image points to file.
            np.savetxt(fn, ipR)

            I += 1

    def write_image_points(self, folderNames):
        """
        Write image points to file system.
        folderNames - A two elements list contains the folder names for the two cameras.

        The self.imagePoints member variable is a 5D array, with the first two dimensions represented
        by Python list. The remaining three dimensions are packed into a NumPy array.
        """

        # Test if the destination directories are present.
        if ( not os.path.isdir(folderNames[G_IDX_L]) ):
            os.makedirs(folderNames[G_IDX_L])
        else:
            # The directory exits. Delete all its content.
            for f in glob.glob( folderNames[G_IDX_L] + '/*.dat' ):
                os.remove(f)
        
        if ( not os.path.isdir(folderNames[G_IDX_R]) ):
            os.makedirs(folderNames[G_IDX_R])
        else:
            # The directory exits. Delete all its content.
            for f in glob.glob( folderNames[G_IDX_R] + '/*.dat' ):
                os.remove(f)

        # Write image points of every camera.
        self.write_image_points_single_camera(folderNames[G_IDX_L], self.imagePoints[G_IDX_L], prefix = PF_IMAGE_POINTS_LEFT)
        self.write_image_points_single_camera(folderNames[G_IDX_R], self.imagePoints[G_IDX_R], prefix = PF_IMAGE_POINTS_RIGHT)

    def find_corners_single_image(self, img, scaleLimit, criteria):
        """
        Find the corners in single image.

        img - The color image which contains a complete checkerboard.
        """

        # The return value.
        cornersSubPix = None

        # Get the gray version of the input image.
        frameGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Loop for every scale.
        for s in range(1, scaleLimit + 1):
            if ( s != 1 ):
                tempGray = cv2.resize( frameGray, (0, 0), fx = s, fy = s, interpolation = cv2.INTER_LINEAR )
            else:
                tempGray = copy.deepcopy(frameGray)

            # Find the corners on the checkerboard.
            ret, corners = cv2.findChessboardCorners(tempGray, self.cbSize,\
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

            if ( ret == True ):
                if ( s != 1 ):
                    # Scale back the corners.
                    corners = corners / s
                    self.showInfo( "Succeed after scaling to %d" % (s) )

                cornersSubPix = cv2.cornerSubPix(frameGray, corners, (11, 11), (-1, -1), criteria)

                break
        
        return cornersSubPix

    def find_corners_single_image_ros(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = downsample_and_detect_corners(img, self.cbSize, self.downsample)

        if ( ret ):
            return corners
        else:
            return None

    def find_corners(self, scaleLimit):
        """
        Find the corners on the images for left and right cameras.
        """

        # Termination criteria.
        criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001 )

        # Clear the data.
        self.imagePoints = []
        self.imagePoints.append([])
        self.imagePoints.append([])

        self.goodImages = []
        self.goodImages.append([])
        self.goodImages.append([])
        self.nGoodImages = 0
        
        self.imageSize = 0
        self.imageHeight = 0
        self.imageWidth  = 0

        # Loop of corner-finding.
        for i in range(self.nImages):
            self.showInfo("Process image pair %d / %d ..." % (i+1, self.nImages), newline = False)

            # Retrieve the file names.
            fileNameL = self.imageFileList[G_IDX_L][i]
            fileNameR = self.imageFileList[G_IDX_R][i]

            # Read the images.
            imgL = cv2.imread(fileNameL)
            imgR = cv2.imread(fileNameR)

            # Check the sizes of these two images.
            if ( 0 == self.imageSize ):
                self.imageSize = imgL.size

            if ( 0 == self.imageHeight or 0 == self.imageWidth):
                self.imageHeight = imgL.shape[0]
                self.imageWidth  = imgL.shape[1]

            if ( imgL.shape[0] != self.imageHeight or imgL.shape[1] != self.imageWidth ):
                return False

            if ( imgL.size != imgR.size ):
                return False
            
            if ( imgL.shape[0] != imgR.shape[0] ):
                return False

            # Sizes are OK. Find corners for each image.
            # cornersL = self.find_corners_single_image( imgL, scaleLimit, criteria )
            cornersL = self.find_corners_single_image_ros( imgL )
            if ( cornersL is None ):
                self.showInfo("Failed to find corners on left camera.")
                continue

            # cornersR = self.find_corners_single_image( imgR, scaleLimit, criteria )
            cornersR = self.find_corners_single_image_ros( imgR )
            if ( cornersR is None ):
                self.showInfo("Failed to find corners on right camera.")
                continue

            self.imagePoints[G_IDX_L].append(cornersL)
            self.imagePoints[G_IDX_R].append(cornersR)

            # Register these good images.
            self.goodImages[G_IDX_L].append(fileNameL)
            self.goodImages[G_IDX_R].append(fileNameR)
            self.nGoodImages += 1

            # Show information.
            self.showInfo("OK.")

        self.write_image_points(self.imagePointsOutDir)

    def check_calibration_quality(self):
        """Check the quality of the calibration."""

        totalImagePoints = 0
        totalError       = 0
        lines            = [[], []]

        for i in range(self.nGoodImages):
            nImagePoint = self.imagePoints[G_IDX_L][i].shape[0]

            # Left camera.
            imgPnt = self.imagePoints[G_IDX_L][i]
            dst    = cv2.undistortPoints(\
                imgPnt, self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
                P = self.cameraMatrix[G_IDX_L])
            line   = cv2.computeCorrespondEpilines( dst, G_IDX_L + 1, self.F )
            lines[G_IDX_L] = line

            # Right camera.
            imgPnt = self.imagePoints[G_IDX_R][i]
            dst    = cv2.undistortPoints(\
                imgPnt, self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
                P = self.cameraMatrix[G_IDX_R])
            line   = cv2.computeCorrespondEpilines( dst, G_IDX_R + 1, self.F )
            lines[G_IDX_R] = line

            for j in range(nImagePoint):
                errIJ = \
                    math.fabs(\
                    self.imagePoints[G_IDX_L][i][j][0][0] * lines[G_IDX_R][j][0][0] +\
                    self.imagePoints[G_IDX_L][i][j][0][1] * lines[G_IDX_R][j][0][1] +\
                    lines[G_IDX_R][j][0][2] ) +\
                    math.fabs(\
                    self.imagePoints[G_IDX_R][i][j][0][0] * lines[G_IDX_L][j][0][0] +\
                    self.imagePoints[G_IDX_R][i][j][0][1] * lines[G_IDX_L][j][0][1] +\
                    lines[G_IDX_L][j][0][2])
                
                totalError += errIJ

            totalImagePoints += nImagePoint
        
        avgError = totalError / totalImagePoints

        return avgError

    def stereo_rectify(self):
        self.isRectified = False

        R1, R2, P1, P2, Q, Roi1, Roi2 = \
        cv2.stereoRectify(\
            self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
            self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
            (self.imageWidth, self.imageHeight),\
            self.R, self.T,\
            alpha = 0, newImageSize = (self.imageWidth, self.imageHeight))
            # flags = cv2.CALIB_ZERO_DISPARITY

        self.isRectified = True

        return R1, R2, P1, P2, Q, Roi1, Roi2

    def compute_remap_and_create_rectified_image(self):
        """
        Comput the remap after calibration and rectification.
        """

        rmap = [[], []]

        # Left camera.
        map1, map2 = cv2.initUndistortRectifyMap(\
            self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
            self.R1, self.P1,\
            ( self.imageWidth, self.imageHeight ), cv2.CV_32FC1)
        rmap[G_IDX_L] = [map1, map2]

        # Right camera.
        map1, map2 = cv2.initUndistortRectifyMap(\
            self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
            self.R2, self.P2,\
            ( self.imageWidth, self.imageHeight ), cv2.CV_32FC1)
        rmap[G_IDX_R] = [map1, map2]

        # Calculate the scaling factor, width and, height of a canvas.
        sf = 1.0 * self.remapCanvasLength / np.max( (self.imageWidth, self.imageHeight) )
        w  = i_r( self.imageWidth  * sf )
        h  = i_r( self.imageHeight * sf )
        canvas = np.zeros( (h, w*2, 3), dtype = np.uint8 )
        path = self.outputPath + "/" + DR_RECTIFIED_IMAGES

        if ( not os.path.isdir(path) ):
            os.makedirs(path)

        # Create rectified images.
        for i in range(self.nGoodImages):
            self.showInfo("Create rectified images (%d / %d)..." % (i+1, self.nGoodImages))

            # Read the images and plot them with overlaying rectangles.
            img  = cv2.imread( self.imageFileList[G_IDX_L][i])
            rimg = cv2.remap( img, rmap[G_IDX_L][0], rmap[G_IDX_L][1], cv2.INTER_LINEAR )

            canvasPartL = cv2.resize(rimg, (w, h), 0, 0, cv2.INTER_AREA)
            cv2.rectangle(canvasPartL,\
                (                  i_r(self.Roi1[0] * sf),                  i_r(self.Roi1[1] * sf) ),\
                ( i_r((self.Roi1[0] + self.Roi1[2]) * sf), i_r((self.Roi1[1] + self.Roi1[3]) * sf) ),\
                RECTIFIED_RECT_COLOR, RECTIFIED_RECT_THICKNESS, 8)

            canvas[:, 0:w, :] = canvasPartL

            img  = cv2.imread( self.imageFileList[G_IDX_R][i])
            rimg = cv2.remap( img, rmap[G_IDX_R][0], rmap[G_IDX_R][1], cv2.INTER_LINEAR )

            canvasPartR = cv2.resize(rimg, (w, h), 0, 0, cv2.INTER_AREA)
            cv2.rectangle(canvasPartR,\
                (                  i_r(self.Roi2[0] * sf),                  i_r(self.Roi2[1] * sf) ),\
                ( i_r((self.Roi2[0] + self.Roi2[2]) * sf), i_r((self.Roi2[1] + self.Roi2[3]) * sf) ),\
                RECTIFIED_RECT_COLOR, RECTIFIED_RECT_THICKNESS, 8)

            canvas[:, w:, :] = canvasPartR

            j = 0
            colorIdx = 0
            while ( j < h ):
                cv2.line(canvas, (0, j), (w*2, j), RECTIFIED_LINE_COLOR_LIST[colorIdx], RECTIFIED_LINE_THICKNESS, 8)
                colorIdx = 1 - colorIdx
                j += 16

            # Save the image.
            outName  = "%s/%04d.jpg" % (path, i)
            outNameL = "%s/Left_%04d.jpg" % (path, i)
            outNameR = "%s/Right_%04d.jpg" % (path, i)
            self.showInfo("Write rectified file %s..." % (outName))
            cv2.imwrite(outName,       canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(outNameL, canvasPartL, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(outNameR, canvasPartR, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #cv2.imshow("Rectified", canvas)

            # wait for the user to visualize the rectified image.
            #cv2.waitKey(500)

    def calibrate(self, scaleLimit = 2):
        """
        scaleLimit - A positive integer indicating the up-scaling factor when trying to find the corners.
        """

        self.isCalibrated = False

        # Test the output directory.
        if ( not os.path.isdir(self.outputPath) ):
            os.makedirs(self.outputPath)

        # Prepare the filenames.
        self.prepare_file_names()

        # Corner finding loop.
        self.find_corners(scaleLimit)

        # Prepare object points.
        objIdx = np.zeros( (self.cbSize[0] * self.cbSize[1], 3), np.float32 )
        objIdx[:, :2] = np.mgrid[0:self.cbSize[0], 0:self.cbSize[1]].T.reshape(-1, 2)*self.cbSquareSize

        # Save objIdx in file.
        np.savetxt(self.outputPath + "/" + FN_OBJECT_INDEX, objIdx)

        self.objectPoints = []

        for i in range(self.nGoodImages):
            self.objectPoints.append(objIdx)

        # Generate initial camera matrices.
        if ( False == self.isIndividuallyCalibrated ):
            self.cameraMatrix = []
            tempCM = cv2.initCameraMatrix2D( self.objectPoints,\
                self.imagePoints[G_IDX_L], ( self.imageWidth, self.imageHeight ), 0 )
            self.cameraMatrix.append(tempCM)

            tempCM = cv2.initCameraMatrix2D( self.objectPoints,\
                self.imagePoints[G_IDX_R], ( self.imageWidth, self.imageHeight ), 0 )
            self.cameraMatrix.append(tempCM)

            self.distortionCoefficients = []
            self.distortionCoefficients.append(None)
            self.distortionCoefficients.append(None)
        else:
            # The cameras are calibrated individually beforehand.
            # Load the camera matrices and distortion coefficients from file system.
            self.load_intrinsics(self.outputPath)

        print( "Initial camera matrix left: " )
        print( self.cameraMatrix[G_IDX_L] )
        print( "" )
        print( "Initial camera matrix right: " )
        print( self.cameraMatrix[G_IDX_R] )
        print( "" )

        # ========== Stereo calibration. ==========

        # Termination criteria.
        criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5 )

        if ( False == self.isIndividuallyCalibrated ):
            ret, self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
                self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
                self.R, self.T, self.E, self.F \
                = cv2.stereoCalibrate(self.objectPoints,\
                self.imagePoints[G_IDX_L], self.imagePoints[G_IDX_R],\
                self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
                self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
                imageSize = (self.imageWidth, self.imageHeight),\
                criteria = criteria,\
                flags = \
                    + cv2.CALIB_USE_INTRINSIC_GUESS \
                    + cv2.CALIB_FIX_K4 \
                    + cv2.CALIB_FIX_K5 \
                    + cv2.CALIB_FIX_K6 )
        else:
            ret, self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
                self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
                self.R, self.T, self.E, self.F \
                = cv2.stereoCalibrate(self.objectPoints,\
                self.imagePoints[G_IDX_L], self.imagePoints[G_IDX_R],\
                self.cameraMatrix[G_IDX_L], self.distortionCoefficients[G_IDX_L],\
                self.cameraMatrix[G_IDX_R], self.distortionCoefficients[G_IDX_R],\
                imageSize = (self.imageWidth, self.imageHeight),\
                criteria = criteria,\
                flags = \
                      cv2.CALIB_FIX_INTRINSIC )

        print_separator("=", 80, "Info.")
        print("Calibration done with rms error =", ret)

        self.isCalibrated = True

        # Check calibration quality.
        avgErr = self.check_calibration_quality()
        print("avgErr = %f." % (avgErr))

        # Stereo rectify.
        self.R1, self.R2, self.P1, self.P2, self.Q,\
        self.Roi1, self.Roi2 = self.stereo_rectify()
        self.write_rectified_matrices()

        # Remap.
        self.compute_remap_and_create_rectified_image()

    def show_calibration_results(self):
        """Show calibration results on the screen."""

        if ( False == self.isCalibrated ):
            print("Not calibrated yet.")
            return

        print( "Camera matrix left: " )
        print( self.cameraMatrix[G_IDX_L] )
        print( "" )
        print( "Camera matrix right: " )
        print( self.cameraMatrix[G_IDX_R] )
        print( "" )

        print( "Distortion matrix left: " )
        print( self.distortionCoefficients[G_IDX_L] )
        print( "" )
        print( "Distortion matrix right: " )
        print( self.distortionCoefficients[G_IDX_R] )
        print( "" )

        print( "R: " )
        print( self.R )
        print( "" )

        print( "T: " )
        print( self.T )
        print( "" )

        print( "E: " )
        print( self.E )
        print( "" )

        print( "F: " )
        print( self.F )
        print( "" )

    def show_rectified_matrices(self):
        if ( self.isRectified == False ):
            print("Not rectified yet.")
            return

        print("R1 = ")
        print(self.R1)

        print("R2 = ")
        print(self.R2)

        print("P1 = ")
        print(self.P1)

        print("P2 = ")
        print(self.P2)

        print("Q = ")
        print(self.Q)

        print("Roi1 = ")
        print(self.Roi1)

        print("Roi2 = ")
        print(self.Roi2)

    def write_calibration_results(self, path = None):
        """Write the calibration results to the folder specified by the path argument."""

        if ( path is None ):
            path = self.outputPath

        if ( False == self.isCalibrated ):
            print( "Not calibrated yet. Will not write results to file." )
            return

        tempPath = path + "/"

        print( "Write image sizes." )
        tempDict = { "width":self.imageWidth, "height":self.imageHeight, "size":self.imageSize }
        self.dump_to_file(tempDict, tempPath + FN_IMAGE_SIZE)

        if ( False == self.isIndividuallyCalibrated ):
            print( "Write camera matrix left." )
            np.savetxt( tempPath + FN_CAMERA_MATRIX_LEFT, self.cameraMatrix[G_IDX_L] )
            print( "Write camera matrix right." )
            np.savetxt( tempPath + FN_CAMERA_MATRIX_RIGHT, self.cameraMatrix[G_IDX_R] )
            
            print( "Write distortion matrix left." )
            np.savetxt( tempPath + FN_DISTORTION_COEFFICIENT_LEFT, self.distortionCoefficients[G_IDX_L] )
            print( "Write distortion matrix right." )
            np.savetxt( tempPath + FN_DISTORTION_COEFFICIENT_RIGHT, self.distortionCoefficients[G_IDX_R] )
        else:
            print(" The cameras are calibrated individually. The camera matrices and distortion coefficients will not be written here. ")

        print( "Write R (rotation matrix)." )
        np.savetxt( tempPath + FN_ROTATION_MATRIX, self.R )

        print( "Write T (translation vector)." )
        np.savetxt( tempPath + FN_TRANSLATION_MATRIX, self.T )

        print( "Write E (essential matrix)." )
        np.savetxt( tempPath + FN_ESSENTIAL_MATRIX, self.E )

        print( "Write F (fundamental matrix)." )
        np.savetxt( tempPath + FN_FUNDAMENTAL_MATRIX, self.F )

        print( "Save the grid size of the calibration board.")
        np.savetxt( tempPath + FN_CB_GRID_SIZE, np.array(self.cbSize, dtype=np.int) )

        print( "Save the square size of the calibration board." )
        np.savetxt( tempPath + FN_CB_SQUARE_SIZE, np.array([self.cbSquareSize], dtype=np.float) )

        print( "Done of writing." )

    def load_intrinsics(self, path):
        # Load camera matrices.
        self.cameraMatrix = []
        self.cameraMatrix.append( np.loadtxt( path + "/" + FN_CAMERA_MATRIX_LEFT,  dtype = np.float ) )
        self.cameraMatrix.append( np.loadtxt( path + "/" + FN_CAMERA_MATRIX_RIGHT, dtype = np.float ) )

        # Load distortion coefficients.
        self.distortionCoefficients = []
        self.distortionCoefficients.append( np.loadtxt( path + "/" + FN_DISTORTION_COEFFICIENT_LEFT,  dtype = np.float ) )
        self.distortionCoefficients.append( np.loadtxt( path + "/" + FN_DISTORTION_COEFFICIENT_RIGHT, dtype = np.float ) )

    def write_rectified_matrices(self, path = None):
        if ( path is None ):
            p = self.outputPath
        else:
            p = path

        if ( False == self.isRectified ):
            print( "Not rectified yet. Will not write results to file." )
            return

        tempPath = p + "/"

        print("Write R1 and R2.")
        np.savetxt(tempPath + FN_R1, self.R1)
        np.savetxt(tempPath + FN_R2, self.R2)

        print("Write P1 and P2.")
        np.savetxt(tempPath + FN_P1, self.P1)
        np.savetxt(tempPath + FN_P2, self.P2)

        print("Write Q.")
        np.savetxt(tempPath + FN_Q, self.Q)

        print("Write ROIs.")
        np.savetxt(tempPath + FN_ROI_1, self.Roi1)
        np.savetxt(tempPath + FN_ROI_2, self.Roi2)

        calib_info = {'k1':self.cameraMatrix[0],
                    'k2':self.cameraMatrix[1],
                    'd1':self.distortionCoefficients[0], 
                    'd2':self.distortionCoefficients[1],
                    'r1':self.R1,
                    'r2':self.R2,
                    'p1':self.P1,
                    'p2':self.P2}

        np.save(tempPath + 'stereo_calib_results.npy', calib_info)


    def load_image_points(self, path, partten = "*.dat"):
        """
        Load image points from path.
        path - The path.
        n - The total number of files to be read from path.
        prefix - The prefix for filenames.
        """

        imgPnt = []

        fileList = sorted( glob.glob(path + "/" + partten) )

        for f in fileList:
            m = np.loadtxt( f )
            shape = m.shape
            m = m.reshape((shape[0], 1, shape[1]))
            imgPnt.append(m)
        
        return imgPnt

    def load_calibration_results(self, path = None):
        """
        Load the calibration results from file system.
        path - The path to find the files. If path is None, self.outputPath will be used.
        """

        if ( path is None ):
            p = self.outputPath
        else:
            p = path

        # Image sizes previously used.
        f = open( p + "/" + FN_IMAGE_SIZE )
        tempDict = json.load(f)
        f.close()
        self.imageWidth  = tempDict["width"]
        self.imageHeight = tempDict["height"]
        self.imageSize   = tempDict["size"]

        # Camera matrices.
        self.cameraMatrix = [[], []]
        self.cameraMatrix[G_IDX_L] = np.loadtxt( p + "/" + FN_CAMERA_MATRIX_LEFT  )
        self.cameraMatrix[G_IDX_R] = np.loadtxt( p + "/" + FN_CAMERA_MATRIX_RIGHT )

        # Distortion coefficients.
        self.distortionCoefficients = [[], []]
        self.distortionCoefficients[G_IDX_L] = np.loadtxt( p + "/" + FN_DISTORTION_COEFFICIENT_LEFT  )
        self.distortionCoefficients[G_IDX_R] = np.loadtxt( p + "/" + FN_DISTORTION_COEFFICIENT_RIGHT )

        # E, F, R, T.
        self.E = np.loadtxt( p + "/" + FN_ESSENTIAL_MATRIX )
        self.F = np.loadtxt( p + "/" + FN_FUNDAMENTAL_MATRIX )
        self.R = np.loadtxt( p + "/" + FN_ROTATION_MATRIX )
        self.T = np.loadtxt( p + "/" + FN_TRANSLATION_MATRIX )

        # Image file lists.
        self.imageFileList = [[], []]

        f = open( p + "/" + FN_IMAGE_FILE_LIST_LEFT )
        self.imageFileList[G_IDX_L] = json.load(f)
        f.close()

        f = open( p + "/" + FN_IMAGE_FILE_LIST_RIGHT )
        self.imageFileList[G_IDX_R] = json.load(f)
        f.close()

        # Image points.
        self.nGoodImages = len(self.imageFileList[0])

        self.imagePoints = [[], []]
        self.imagePoints[G_IDX_L] = self.load_image_points(p + "/" + DR_IMAGE_POINTS_LEFT)
        self.imagePoints[G_IDX_R] = self.load_image_points(p + "/" + DR_IMAGE_POINTS_RIGHT)

        self.isCalibrated = True
