from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import logging
from utils.annolist import AnnotationLib as AnnoLib
from utils.data_utils import annotation_jitter, annotation_to_h5
import tensorflow as tf
import numpy as np
import scipy
import itertools
import random
import threading
from collections import namedtuple


CLASSES = ['car', 'pedestrian', 'cyclist', 'traffic_light', 'traffic_sign', 'truck']
CLASS_IDS = { CLASSES[x]: x for x in range(len(CLASSES))}

def read_bdd_anno(label_file):
    """Reads a BDD100K annotation file (in kitti format)
    
    Arguments:
        label_file {str} -- Filepath to the label file
    Returns:
        List of bboxes with associated class id
    """
    logging.info("KittiBox bdd_input.py: Reading label file {}".format(label_file))
    
    return extract_bboxes(label_file)


def extract_bboxes(label_file):
    """ 
    Extracts all values of interest from the bdd kitti label file and returns a list where
    each row is a bounding box for an object in the image.
    """
    bboxes = []
    labels = _gen_label_list(label_file)
    for label in labels:
        logging.info("Reading label" , label)
        if label[0] not in CLASSES:
            logging.warn("{} not in classes - skipping".format(label[0]))
            continue
        bbox_rect = extract_bbox_rect(label)
        bbox_rect.classID = CLASSES.index(label[0])
        bboxes.append(bbox_rect)
    return bboxes


def extract_bbox_rect(label_list):
    """ Returns an AnnoRect object from the data in the given label list """
    x1, y1, x2, y2 = [float(x) for x in label_list[4:8]]
    if x1 > x2 or y1 > y2:
        logging.warn("Bounding boxes may have illegal format -> x1,y1,x2,y2: ({},{},{},{})".format(x1,y1,x2,y2))
        raise ValueError
    return AnnoLib.AnnoRect(x1=x1, y1=y1, x2=x2, y2=y2)


def _gen_label_list(label_file):
    """ Generates a list where each entry is a list in the kitti-data format (see kitti_format.md for
    more details).
    """
    label_lines = open(label_file).readlines()

    label_list = []
    for label_row in label_lines:
        #logging.info(f"Parsing label line {label_row} in {label_file}")
        labels = label_row.split(' ')
        assert len(labels) == 15, "Expected number of columns to be 15, not" + str(len(labels))
        label_list.append(labels)
    return label_list



def _augment_image(hypes, image):
    """Augments the given image by randomly applying brightness, contras, saturation and hue."""
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    logging.info("Augmenting image in KittiBox")
    augment_level = hypes['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    if augment_level > 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)
        image = tf.image.random_hue(image, max_delta=0.15)

    image = tf.minimum(image, 255.0)
    image = tf.maximum(image, 0)
    return image

def _create_bdd_annotations(image_file, gt_image_file):
    assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file

    assert os.path.exists(gt_image_file), \
        "File does not exist: %s" % gt_image_file

    rect_list = read_bdd_anno(gt_image_file)

    anno = AnnoLib.Annotation()
    anno.rects = rect_list
    return anno

def _load_and_resize_image(image_file, anno, hypes):
    """ Loads an image from the image file, and resizes the image and bounding boxes if necessary"""
    im = scipy.misc.imread(image_file)
    if im.shape[2] == 4:
        im = im[:, :, :3]
    if not _is_correct_img_size(im, hypes):
        anno = _rescale_boxes(im.shape, anno,
                                hypes["image_height"],
                                hypes["image_width"])
        im = scipy.misc.imresize(
            im, (hypes["image_height"], hypes["image_width"]),
            interp='cubic')
    return im

def _apply_jitter(im, anno, hypes):
    logging.info("Applying jitter to image (kittibox)")
    jitter_scale_min = 0.9
    jitter_scale_max = 1.1
    jitter_offset = 16
    im, anno = annotation_jitter(
        im, anno, target_width=hypes["image_width"],
        target_height=hypes["image_height"],
        jitter_scale_min=jitter_scale_min,
        jitter_scale_max=jitter_scale_max,
        jitter_offset=jitter_offset)
    return im, anno


def _load_bdd_txt(bdd_txt, hypes, jitter=False, random_shuffle=True):
    """Loads the given bdd txt file and outputs 
    
    Arguments:
        bdd_txt -- A .txt file with all pairs of images and masks
    
    Keyword Arguments:
        jitter {bool} -- Whether or not to apply jittering on the images (default: {False})
        random_shuffle {bool} -- Whether or not to yield images in random order (default: {True})
    """
    
    base_path = os.path.realpath(os.path.dirname(bdd_txt))
    files = [line.rstrip() for line in open(bdd_txt)]
    if hypes['data']['truncate_data']:
        files = files[:10]
        random.seed(0)

    for epoch in itertools.count():
        if random_shuffle:
            random.shuffle(files)
        # Iterate through all files, optionally applying jitter, yielding images masks and bboxes for each
        for file in files:
            image_file, gt_image_file = file.split(" ")
            #image_file = os.path.join(base_path, image_file)
            #gt_image_file = os.path.join(base_path, gt_image_file)

            logging.info("Creating bdd annotations for im: {}, gt: {}".format(image_file, gt_image_file))
            anno = _create_bdd_annotations(image_file, gt_image_file)

            im = _load_and_resize_image(image_file, anno, hypes)

            if jitter:
                im, anno = _apply_jitter(im, anno, hypes)

            pos_list = [rect for rect in anno.rects if rect.classID == 1]

            fake_anno = namedtuple('fake_anno_object', ['rects'])
            pos_anno = fake_anno(pos_list)

            boxes, confs = annotation_to_h5(hypes,
                                            pos_anno,
                                            hypes["grid_width"],
                                            hypes["grid_height"],
                                            hypes["rnn_len"])

            # BDD100K does not contain any 'DontCare' areas
            mask_list = [rect for rect in anno.rects if rect.classID == -1]
            mask = _generate_mask(hypes, mask_list)
            grid_width = hypes["grid_width"]
            grid_height = hypes["grid_height"]
            #mask = np.ones([grid_height, grid_width])

            boxes = boxes.reshape([grid_height,
                                   grid_width, 4])
            confs = confs.reshape(grid_height, grid_width)

            yield {"image": im, "boxes": boxes, "confs": confs,
                   "rects": pos_list, "mask": mask}


def _is_correct_img_size(img, hypes):
    """ Returns True if the image is conforming with requirements of
    image height and width defined in hypes
    """
    return  img.shape[0] == hypes["image_height"] and img.shape[1] == hypes["image_width"]


def _generate_mask(hypes, ignore_rects):

    width = hypes["image_width"]
    height = hypes["image_height"]
    grid_width = hypes["grid_width"]
    grid_height = hypes["grid_height"]

    mask = np.ones([grid_height, grid_width])

    if not hypes['use_mask']:
        return mask

    for rect in ignore_rects:
        left = int((rect.x1+2)/width*grid_width)
        right = int((rect.x2-2)/width*grid_width)
        top = int((rect.y1+2)/height*grid_height)
        bottom = int((rect.y2-2)/height*grid_height)
        for x in range(left, right+1):
            for y in range(top, bottom+1):
                mask[y, x] = 0

    return mask


def start_enqueuing_threads(hypes, q, phase, sess):
    # Creating Placeholder for the Queue
    x_in = tf.placeholder(tf.float32)
    confs_in = tf.placeholder(tf.float32)
    boxes_in = tf.placeholder(tf.float32)
    mask_in = tf.placeholder(tf.float32)

    enqueue_op = q.enqueue((x_in, confs_in, boxes_in, mask_in))

    def make_feed(data):
        return {x_in: data['image'],
                confs_in: data['confs'],
                boxes_in: data['boxes'],
                mask_in: data['mask']}
    def thread_loop(sess, enqueue_op, gen):
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    data_file = hypes["data"]['%s_file' % phase]
    data_dir = hypes['dirs']['data_dir']
    data_file = os.path.join(data_dir, data_file)

    gen = _load_bdd_txt(data_file, hypes,
                          jitter={'train': hypes['solver']['use_jitter'],
                                  'val': False}[phase])
    data = next(gen)
    sess.run(enqueue_op, feed_dict=make_feed(data))
    t = threading.Thread(target=thread_loop,
                         args=(sess, enqueue_op, gen))
    t.daemon = True
    t.start()


def create_queues(hypes, phase):
    """Create Queues."""
    logging.info("Creating queues in bdd_inputs (KittiBox)")
    hypes["rnn_len"] = 1
    dtypes = [tf.float32, tf.float32, tf.float32, tf.float32]
    grid_size = hypes['grid_width'] * hypes['grid_height']
    shapes = ([hypes['image_height'], hypes['image_width'], 3],
              [hypes['grid_height'], hypes['grid_width']],
              [hypes['grid_height'], hypes['grid_width'], 4],
              [hypes['grid_height'], hypes['grid_width']])
    capacity = 30
    q = tf.FIFOQueue(capacity=capacity, dtypes=dtypes, shapes=shapes)
    return q


def inputs(hypes, q, phase):

    if phase == 'val':
        logging.info("Entering val phase in bdd_inputs.py (KittiBox)")
        image, confidences, boxes, mask = q.dequeue()
        image = tf.expand_dims(image, 0)
        confidences = tf.expand_dims(confidences, 0)
        boxes = tf.expand_dims(boxes, 0)
        mask = tf.expand_dims(mask, 0)
        return image, (confidences, boxes, mask)
    elif phase == 'train':
        logging.info("Entering train phase in bdd_inputs.py (KittiBox)")
        image, confidences, boxes, mask = q.dequeue_many(hypes['batch_size'])
        image = _augment_image(hypes, image)
        return image, (confidences, boxes, mask)
    else:
        assert("Bad phase: {}".format(phase))

def _rescale_boxes(current_shape, anno, target_height, target_width):
    logging.info("Rescaling box {} to ({}, {})".format(current_shape, target_height, target_width))
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.x1 < r.x2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return anno
