import tensorflow as tf
from imports import *

def calculate_IOU(b1, b2):
    zero = tf.convert_to_tensor(0, b1.dtype)
    b1_x1, b1_y1, b1_x2, b1_y2 = tf.unstack(b1, 4, axis=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = tf.unstack(b2, 4, axis=1)

    b1_width = tf.maximum(zero, b1_x2 - b1_x1)
    b1_height = tf.maximum(zero, b1_y2 - b1_y1)
    b2_width = tf.maximum(zero, b2_x2 - b2_x1)
    b2_height = tf.maximum(zero, b2_y2 - b2_y1)

    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_x1 = tf.maximum(b1_x1, b2_x1)
    intersect_y1 = tf.maximum(b1_y1, b2_y1)

    intersect_x2 = tf.maximum(b1_x2, b2_x2)
    intersect_y2 = tf.maximum(b1_y2, b2_y2)

    intersect_width = tf.maximum(zero, intersect_x2 - intersect_x1)
    intersect_height = tf.maximum(zero, intersect_y2, intersect_y1)

    intersect_area = intersect_width * intersect_height
    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    return iou

def IOU(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    iou = calculate_IOU(y_pred, y_true)
    return iou

def evaluate(actual, pred):
    iou = IOU(actual, pred)
    loss = losses.MSE(actual, pred)
    return loss, iou

criteron = evaluate