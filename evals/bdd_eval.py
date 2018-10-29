#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the MediSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import scipy as scp
import scipy.misc

import numpy as np

import tensorflow as tf

import utils.train_utils
import time

import random

from utils.annolist import AnnotationLib as AnnLib

import logging
import sys

def make_val_dir(hypes, validation=True):
    if validation:
        val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_out')
    else:
        val_dir = os.path.join(hypes['dirs']['output_dir'], 'train_out')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def make_img_dir(hypes):
    val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_images')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def write_rects(rects, filename):
    with open(filename, 'w') as f:
        for rect in rects:
            string = "Car 0 1 0 %f %f %f %f 0 0 0 0 0 0 0 %f" % \
                (rect.x1, rect.y1, rect.x2, rect.y2, rect.score)
            print(string, file=f)


def evaluate(hypes, sess, image_pl, softmax):
    logging.info("KittiBox: Evaluating images and bboxes")
    pred_annolist, image_list, dt, dt2 = get_results(
        hypes, sess, image_pl, softmax, True)

    val_path = make_val_dir(hypes)

    eval_list = []

    eval_cmd = os.path.join(hypes['dirs']['base_path'],
                            hypes['data']['eval_cmd'])

    label_dir = os.path.join(hypes['dirs']['data_dir'],
                             hypes['data']['label_dir'])

    logging.info("Trying to call kitti evaluation code - eval_cmd: {}, val_path: {}, label_dir: {}".format(eval_cmd, val_path, label_dir))
    try:
        cmd = [eval_cmd, val_path, label_dir]
        popen = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True).communicate()
        
    except OSError as error:
        logging.warning("Failed to run run kitti evaluation code.")
        logging.warning("Please run: `cd submodules/KittiObjective2/ && make`")
        logging.warning("For more information see:"
                        "`submodules/KittiObjective2/README.md`")
        exit(1)
        img_dir = make_img_dir(hypes)
        logging.info("Output images have been written to {}.".format(img_dir))
        eval_list.append(('Speed (msec)', 1000*dt))
        eval_list.append(('Speed (fps)', 1/dt))
        eval_list.append(('Post (msec)', 1000*dt2))
        return eval_list, image_list

    res_file = os.path.join(val_path, "stats_car_detection.txt")
    logging.info("Trying to evaluate stats file {}".format(res_file))
    with open(res_file) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            eval_list.append(("val   " + mode, mean))

    pred_annolist, image_list2, dt, dt2 = get_results(
        hypes, sess, image_pl, softmax, False)

    val_path = make_val_dir(hypes, False)
    
    subprocess.check_call([eval_cmd, val_path, label_dir], stdout=sys.stdout)
    res_file = os.path.join(val_path, "stats_car_detection.txt")

    with open(res_file) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            eval_list.append(("train   " + mode, mean))

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))
    eval_list.append(('Post (msec)', 1000*dt2))

    return eval_list, image_list


def get_results(hypes, sess, image_pl, decoded_logits, validation=True):

    if hypes['use_rezoom']:
        pred_boxes = decoded_logits['pred_boxes_new']
    else:
        pred_boxes = decoded_logits['pred_boxes']
    pred_confidences = decoded_logits['pred_confidences']

    # Build Placeholder
    shape = [hypes['image_height'], hypes['image_width'], 3]

    if validation:
        kitti_txt = os.path.join(hypes['dirs']['data_dir'],
                                 hypes['data']['val_file'])
    else:
        kitti_txt = os.path.join(hypes['dirs']['data_dir'],
                                 hypes['data']['train_file'])
    # true_annolist = AnnLib.parse(test_idl)

    val_dir = make_val_dir(hypes, validation)
    img_dir = make_img_dir(hypes)

    image_list = []

    pred_annolist = AnnLib.AnnoList()

    files = [line.rstrip() for line in open(kitti_txt)]
    base_path = os.path.realpath(os.path.dirname(kitti_txt))

    for i, file in enumerate(files):
        image_file = file.split(" ")[0]
        if not validation and random.random() > 0.2:
            continue
        #image_file = os.path.join(base_path, image_file)
        orig_img = scp.misc.imread(image_file)[:, :, :3]
        img = scp.misc.imresize(orig_img, (hypes["image_height"],
                                           hypes["image_width"]),
                                interp='cubic')
        feed = {image_pl: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)
        pred_anno = AnnLib.Annotation()
        pred_anno.imageName = image_file
        new_img, rects = utils.train_utils.add_rectangles(
            hypes, [img], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.50, tau=hypes['tau'], color_acc=(0, 255, 0))

        if validation and i % 15 == 0:
            image_name = os.path.basename(pred_anno.imageName)
            image_name = os.path.join(img_dir, image_name)
            scp.misc.imsave(image_name, new_img)

        if validation:
            image_name = os.path.basename(pred_anno.imageName)
            image_list.append((image_name, new_img))
        # get name of file to write to
        image_name = os.path.basename(image_file)
        val_file_name = image_name.split('.')[0] + '.txt'
        val_file = os.path.join(val_dir, val_file_name)

        # write rects to file

        pred_anno.rects = rects
        pred_anno = utils.train_utils.rescale_boxes((
            hypes["image_height"],
            hypes["image_width"]),
            pred_anno, orig_img.shape[0],
            orig_img.shape[1])

        write_rects(rects, val_file)

        pred_annolist.append(pred_anno)

    start_time = time.time()
    for i in xrange(100):
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)
    dt = (time.time() - start_time)/100

    start_time = time.time()
    for i in xrange(100):
        utils.train_utils.compute_rectangels(
            hypes, np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.001, tau=hypes['tau'])
    dt2 = (time.time() - start_time)/100

    return pred_annolist, image_list, dt, dt2
