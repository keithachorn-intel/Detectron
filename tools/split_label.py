from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

import feather
import pandas as pd
import numpy as np
import pycocotools.mask as mask_util
from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
from utils.colormap import colormap

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

from video_label import label_video, init_model

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--cfg', dest='cfg', default=None, type=str,
        help='cfg model file (/path/to/model_config.yaml)')
    parser.add_argument('--wts', dest='weights', default=None, type=str,
        help='weights model file (/path/to/model_weights.pkl)')
    parser.add_argument('--video_dir', required=True, type=str)
    parser.add_argument('--feather_dir', required=True, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    model = init_model(args.cfg, args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    for vid_fname in os.listdir(args.video_dir):
        vid_id = int(vid_fname.split('.')[0])
        if vid_id % 2500 != 0:
            continue
        vid_fname = os.path.join(args.video_dir, vid_fname)
        feather_fname = os.path.join(args.feather_dir, str(vid_id) + '.feather')
        print(vid_id, vid_fname, feather_fname)
        cap = cv2.VideoCapture(vid_fname)

        label_video(model, dummy_coco_dataset, cap, feather_fname)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
