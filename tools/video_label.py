#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

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

import swag

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

# schema:
# 'frame', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'

def get_rows(frame, boxes, segms=None, thresh=0.05, dataset=None):
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
            boxes, segms, None)

    rows = []
    if boxes is None:
        return rows
    for i in range(len(boxes)):
        score = boxes[i, -1]
        if score < thresh:
            continue
        bbox = boxes[i, :4]
        cls = classes[i]
        xmin, ymin, xmax, ymax = bbox
        row = [frame, cls, score, xmin, ymin, xmax, ymax]
        rows.append(row)
    return rows

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--cfg', dest='cfg', default=None, type=str,
        help='cfg model file (/path/to/model_config.yaml)')
    parser.add_argument('--wts', dest='weights', default=None, type=str,
        help='weights model file (/path/to/model_weights.pkl)')
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--nb_frames', default=10000000, type=int)
    parser.add_argument('--video_fname', required=True, type=str)
    parser.add_argument('--index_fname', type=str)
    parser.add_argument('--feather_fname', required=True, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def label_video(model, dataset, cap, feather_fname,
                nb_frames=100000000, start_frame=0):
    logger = logging.getLogger(__name__)

    all_rows = []
    for i in range(nb_frames):
        frame = start_frame + i
        ret, im = cap.read()
        if not ret:
            break
        logger.info('Processing frame %d' % i)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers)
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        rows = get_rows(frame, cls_boxes, segms=cls_segms, dataset=dataset)
        all_rows += rows

    df = pd.DataFrame(all_rows,
                      columns=['frame', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
    f32 = ['conf', 'xmin', 'ymin', 'xmax', 'ymax']
    for f in f32:
        df[f] = df[f].astype('float32')
    df['frame'] = df['frame'].astype('int32')
    df['cname'] = df['cname'].astype('int8')
    df = df.sort_values(by=['frame', 'cname', 'conf'], ascending=[True, True, False])
    print(df)
    feather.write_dataframe(df, feather_fname)

    cap.release()

def main(args):
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    # No mask, no keypoints
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINTS_ON = False
    cfg.TEST.BBOX_AUG.ENABLED = False
    # Soft NMS
    # cfg.TEST.SOFT_NMS.ENABLED = True
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    USE_SWAG = True
    if USE_SWAG:
        cap = swag.VideoCapture(args.video_fname, args.index_fname)
    else:
        cap = cv2.VideoCapture(args.video_fname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    label_video(model, dummy_coco_dataset, cap, args.feather_fname,
                args.nb_frames, args.start_frame)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
