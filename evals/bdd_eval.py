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
from utils.train_utils import CLASS_COLORS
from utils.annolist import AnnotationLib as AnnLib

import random
import logging
import sys
import time

CLASSES = ['Car', 'Person', 'Bike', 'Traffic_light', 'Traffic_sign', 'Truck']

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


def calc_fps(sess, hypes, pred_boxes, pred_confidences, feed):
    logging.info("Starting to evaluate fps of network")
    start_time = time.time()
    for i in range(100):
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)
    dt = (time.time() - start_time)/100

    start_time = time.time()
    for i in range(100):
        utils.train_utils.compute_rectangels(
            hypes, np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.001, tau=hypes['tau'])
    dt2 = (time.time() - start_time)/100
    return dt, dt2


def write_rects(rects, filename):
    # TODO: Expand this to write bboxes of different classes
    out_str = []
    with open(filename, 'w') as f:
        for rect in rects:
            class_idx = rect.classID
            # By default, assume traffic light if class id not assigned (but this should not happen)
            if class_idx is None:
                logging.warn("Found rect without class id in evals.py")
                class_idx = 3
            class_str = CLASSES[class_idx - 1]
            string = "%s 0 1 0 %f %f %f %f 0 0 0 0 0 0 0 %f" % \
                (class_str, rect.x1, rect.y1, rect.x2, rect.y2, rect.score)
            print(string, file=f)
            out_str.append(string)
    if out_str:
        logging.info("Writing ***** \n {} \n ****** to file {}\n".format("\n".join(out_str), filename))


def draw_rects(image, rects, color=(255, 192, 203)):
    """ Returns the image with all rectangles drawn over it
        Default color is pink, else uses class colors from train utils.
    """ 
    from PIL import Image, ImageDraw
    poly = Image.new('RGBA', image.size)
    pdraw = ImageDraw.Draw(poly)
    for rect in rects:
        rect_cords = ((rect.x1, rect.y1), (rect.x1, rect.y2),
                      (rect.x2, rect.y2), (rect.x2, rect.y1),
                      (rect.x1, rect.y1))
        if rects.class_id is not None:
            color = CLASS_COLORS[rects.class_id]
        else:
            logging.warn("Could not find class id for rect - drawing as default color {}".format(color))
        pdraw.line(rect_cords, fill=color, width=2)
    image.paste(poly, mask=poly)
    return np.array(image)


def create_eval_stats(detection_filepath, phase="train"):
    """ Takes as input a folder where the detection stats of the different 
        classes should be, and output the mean detection stats on 
        easy, medium and hard difficulties.  
        Returns: A list of detection stats where each element is a tuple 
                 consisting of (phase + mode, mean)

    """
    # TODO: This should probably not return evaluation for each class, but a mean of 
    # the detection results.
    eval_list = []
    for class_name in CLASSES:
        filename = "stats_{}_detection.txt".format(class_name.lower())
        res_file = os.path.join(detection_filepath, filename)
        logging.info("Trying to evaluate stats file {}".format(res_file))
        # Skip classes that have no detections
        if not os.path.isfile(res_file):
            logging.warn("There are no detections for class {}".format(class_name))
            continue
        with open(res_file) as f:
            for mode in ['easy', 'medium', 'hard']:
                line = f.readline()
                if not line:
                    continue
                #logging.info("Reading line: {}".format(line))
                result = np.array(line.rstrip().split(" ")).astype(float)
                mean = np.mean(result)
                eval_list.append(("{}: {}   {}".format(class_name, phase, mode), mean))
    return eval_list


def call_kitti_eval_code(eval_cmd, val_path, label_dir):
    """ Attempts to call the KittiObjective evaluation code """ 
    try:
        # Call eval command with val_path (val_out)
        cmd = [eval_cmd, val_path, label_dir]
        popen = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True).communicate()
    except OSError as error:
        logging.warning("Failed to run run kitti evaluation code.")
        logging.warning("Please run: `cd submodules/KittiObjective2/ && make`")
        logging.warning("For more information see:"
                        "`submodules/KittiObjective2/README.md`")
        exit(1)


def evaluate(hypes, sess, image_pl, softmax):
    logging.info("KittiBox: Evaluating images and bboxes")
    pred_annolist, image_list, dt, dt2 = get_results(
        hypes, sess, image_pl, softmax, True)


    eval_list = []
    eval_cmd = os.path.join(hypes['dirs']['base_path'],
                            hypes['data']['eval_cmd'])
    label_dir = os.path.join(hypes['dirs']['data_dir'],
                             hypes['data']['label_dir'])

    # Call kitti objective code on validation
    val_path = make_val_dir(hypes, validation=True)
    logging.info("Trying to call kitti evaluation code - eval_cmd: {}, val_path: {}, label_dir: {}".format(eval_cmd, val_path, label_dir))
    call_kitti_eval_code(eval_cmd, val_path, label_dir)
    
    # Get evaluation of detections on validation
    val_detection_results = create_eval_stats(val_path, phase="val")
    eval_list.extend(val_detection_results)

    pred_annolist, image_list2, dt, dt2 = get_results(hypes, sess, image_pl, softmax, validation=False)
    
    # Call kitti objective code on train
    val_path = make_val_dir(hypes, validation=False)
    logging.info(
        "Trying to call kitti evaluation code (pt2) - eval_cmd: {}, val_path: {}, label_dir: {}".format(eval_cmd, val_path, label_dir))
    call_kitti_eval_code(eval_cmd, val_path, label_dir)
    
    # Get evaluation of detections on train
    train_detection_results = create_eval_stats(val_path, phase="train")
    eval_list.extend(train_detection_results)    

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))
    eval_list.append(('Post (msec)', 1000*dt2))

    return eval_list, image_list


def get_results(hypes, sess, image_pl, decoded_logits, validation=True):
    """ Evaluates detection results and prints the evaluation results for all images in the chosen phase  """
    logging.info("get_results(): Called with image pl {} and validation={}".format(image_pl, validation))
    if hypes['use_rezoom']:
        pred_boxes = decoded_logits['pred_boxes_new']
    else:
        pred_boxes = decoded_logits['pred_boxes']
    pred_confidences = decoded_logits['pred_confidences']

    # Build Placeholder for image
    shape = [hypes['image_height'], hypes['image_width'], 3]

    if validation:
        kitti_txt = os.path.join(hypes['dirs']['data_dir'], hypes['data']['val_file'])
    else:
        kitti_txt = os.path.join(hypes['dirs']['data_dir'], hypes['data']['train_file'])
    logging.info("get_results() validation: {}: Trying to test  with kitti-predictions on path {}".format(validation, kitti_txt))

    files = [line.rstrip() for line in open(kitti_txt)]

    pred_annolist, image_list, feed = run_session(hypes, sess, files, pred_boxes, pred_confidences, image_pl, validation)

    dt, dt2 = calc_fps(sess, hypes, pred_boxes, pred_confidences, feed)
    return pred_annolist, image_list, dt, dt2



def run_session(hypes, sess, files, pred_boxes, pred_confidences, image_pl, validation):
    """ Accepts as input a list of files and placeholders.
	Then, runs the network to get predicted confidences and boxes for each file.
        Returns a image feed, image list and pred annolist
    """
    val_dir = make_val_dir(hypes, validation)
    img_dir = make_img_dir(hypes)
    #base_path = os.path.realpath(os.path.dirname(kitti_txt))
    base_path = "/notebooks"
    pred_annolist = AnnLib.AnnoList()
    image_list = []
    for i, file in enumerate(files):
        image_file = file.split(" ")[0]
        if not validation and random.random() > 0.2:
            continue

        image_file = os.path.join(base_path, image_file)
        orig_img = scp.misc.imread(image_file)[:, :, :3]
        img = scp.misc.imresize(orig_img, (hypes["image_height"],
                                           hypes["image_width"]),
                                interp='cubic')
        feed = {image_pl: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)

        new_img, rects = utils.train_utils.add_rectangles(
            hypes, [img], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.50, tau=hypes['tau'], color_acc=(0, 255, 0))

        if validation and i % 15 == 0:
            image_name = os.path.basename(image_file)
            image_name = os.path.join(img_dir, image_name)
            if i % 15 == 0:
                scp.misc.imsave(image_name, new_img)
 
        if rects:
            logging.info("All rects (bdd_eval): {}".format(rects))

        # get name of file to write rectangles to to
        image_name = os.path.basename(image_file)
        val_file_name = image_name.split('.')[0] + '.txt'
        val_file = os.path.join(val_dir, val_file_name)
        logging.info("KittiBox: Name of val file: {}".format(val_file))
        # Write all rectangles (detections)
        write_rects(rects, val_file)
        pred_anno = create_and_rescale_annos(hypes, image_file, rects, orig_img)
        pred_annolist.append(pred_anno)

    if feed is None:
        logging.warn("Feed is none - available files are {}".format(files))
        image_file = os.path.join(base_path, image_file)
        orig_img = scp.misc.imread(image_file)[:, :, :3]
        img = scp.misc.imresize(orig_img, (hypes["image_height"],
                                           hypes["image_width"]),
                                interp='cubic')
        feed = {image_pl: img}
    return pred_annolist, image_list, feed


def create_and_rescale_annos(hypes, image_file, rects, orig_img):
    """ Creates an annotation object with rects. Then rescales the boxes to correspond to image shape
        Returns a AnnLib.Annotation object
    """
    pred_anno = AnnLib.Annotation()
    pred_anno.imageName = image_file
    pred_anno.rects = rects
    pred_anno = utils.train_utils.rescale_boxes((
            hypes["image_height"],
            hypes["image_width"]),
            pred_anno, orig_img.shape[0],
            orig_img.shape[1])
    return pred_anno

