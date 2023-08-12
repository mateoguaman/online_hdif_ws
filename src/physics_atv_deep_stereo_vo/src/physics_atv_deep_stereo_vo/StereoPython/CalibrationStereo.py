from __future__ import print_function

import argparse
import cv2
import glob
import numpy as np
import os

from StereoCalibrator import StereoCalibrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate stereo camera.")

    parser.add_argument("basedir", type=str, \
        help="The base directory.")
    
    parser.add_argument("outdir", type=str, \
        help="The sub-directory for the output results.")

    parser.add_argument("--row", type=int, default=6, \
        help="The number of row of corners on the chessboard.")

    parser.add_argument("--col", type=int, default=8, \
        help="The number of column of corners on the chessboard.")

    parser.add_argument("--csize", type=float, default=0.1185, \
        help="The width of the squares on the chessboard. Unit m.")

    parser.add_argument("--image-pattern", type=str, default="*.png", \
        help="The file search pattern for the input images.")

    parser.add_argument("--individually", action="store_true", default=False, \
        help="Set this flag to perform calibraion with the two cameras pre-calibrated individually. The pre-calibrated intrinsics must be present in the output directory.")
    
    parser.add_argument("--downsample", type=int, default="307200", 
        help="The target downsample pixel numbers. Default number comes from 640x480.")

    args = parser.parse_args()

    print("Begin calibrating %s. " % ( args.basedir ))

    leftDir  = "%s/%s" % ( args.basedir, "left" )
    rightDir = "%s/%s" % ( args.basedir, "right" )

    calib = StereoCalibrator(\
        (args.col, args.row),\
         args.csize, leftDir, rightDir, args.outdir,
         args.downsample)

    calib.isDebugging = False
    calib.debuggingNImagePairs = 20

    calib.filenamePattern = args.image_pattern
    calib.isIndividuallyCalibrated = args.individually

    calib.calibrate()
    calib.show_calibration_results()
    calib.write_calibration_results(args.outdir)
