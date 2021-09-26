# -*- coding:utf-8 -*-
import copy
from typing import List, Tuple, Dict

from tensorflow.lite.python.interpreter import Interpreter

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

TYPE_INT8 = "INT8"
TYPE_FLOAT16 = "FLOAT16"

MODEL_TYPE = (TYPE_INT8, TYPE_FLOAT16)
MODEL_PATH = {TYPE_INT8: "weights/model_int8.tflite",
              TYPE_FLOAT16: "weights/model_fp16.tflite"}


def detect(image: np.ndarray, num_threads: int, model_type: str, confidence_thr: float) -> Tuple[List, List, List, int]:
    """do object detection

    Args:
        image (np.ndarray): input image
        num_threads (int): number of tflite threads
        model_type (str): model type
        confidence_thr (float): threshold

    Returns:
        Tuple[List, List, List, int]: [bboxes, classes, scores, num_of_detection]
    """
    @st.cache(allow_output_mutation=True)
    def load(num_threads: int, model_type: str) -> Tuple[Interpreter, Tuple, Dict]:
        """load tflie model

        Args:
            num_threads (int): number of tflite thread
            model_type (str): model type

        Returns:
            Tuple[Interpreter, Tuple, Dict]: [tflite interpreter, image size, label_map]
        """
        print(
            f"LOAD MODEL ... thread: {num_threads}, model_type : {model_type}")
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH[model_type], num_threads=num_threads)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        img_size = (
            input_details['shape'][1], input_details['shape'][2])

        label_map = {}
        with open("weights/labelmap.txt", 'r') as f:
            for i, line in enumerate(f):
                label_map[i] = line.strip()
        return interpreter, img_size, label_map

    interpreter, img_size, label_map = load(num_threads, model_type)

    # # pre-process
    img, width, height = _pre_process(image, img_size, model_type)

    # predict
    bboxes, classes, scores, detect_num = _predict(img, interpreter)

    # post-process
    bboxes, classes, scores, detect_num = _post_process(
        width, height, bboxes, classes, scores, detect_num, label_map, confidence_thr)

    return bboxes, classes, scores, detect_num


def _pre_process(img: np.ndarray, img_size: Tuple, model_type: str) -> Tuple[np.ndarray, int, int]:
    """pre-process

    Args:
        img (np.ndarray): input image
        img_size (Tuple): target image size
        model_type (str): model type

    Returns:
        Tuple[np.ndarray, int, int]: [preprocessed image, original width, original height]
    """
    width, height = img.shape[1], img.shape[0]
    img = cv2.resize(img, img_size)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    if model_type == TYPE_INT8:
        img = img.astype(np.uint8)
    elif model_type == TYPE_FLOAT16:
        img = img.astype(np.float32)
    return img, width, height


def _predict(img: np.array, interpreter) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """predict

    Args:
        img (np.array): image
        interpreter ([type]): tflite model

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]: [bboxes, classes, scores, num_of_detection]
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details['index'], img)
    interpreter.invoke()

    bboxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])
    return np.squeeze(bboxes), np.squeeze(classes), np.squeeze(scores), int(num[0])


def _post_process(width: int, height: int, bboxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, object_num: int, label_map: dict, thr: float) -> Tuple[List, List, List, int]:
    """post-process

    Args:
        width (int): original image width
        height (int): original image height
        bboxes (np.ndarray): detected bbox
        classes (np.ndarray): class names
        scores (np.ndarray): scores
        object_num (int): num_of_detection
        label_map (dict): label map
        thr (float): threshold

    Returns:
        Tuple[List, List, List, int]: [bbox, classes, scores, num_of_detection]
    """
    post_bboxes, post_classes, post_scores = [], [], []

    for i in range(object_num):
        if float(scores[i]) > thr:
            post_classes.append(label_map[int(classes[i])])
            post_scores.append(float(scores[i]))

            x1, y1 = int(bboxes[i][1] * width), int(bboxes[i][0] * height)
            x2, y2 = int(bboxes[i][3] * width), int(bboxes[i][2] * height)
            post_bboxes.append([x1, y1, x2, y2])

    return post_bboxes, post_classes, post_scores, len(post_scores)


def draw_bboxes(image: np.ndarray, bboxes: List, scores: List, classes: List, detect_num: int) -> np.ndarray:
    """draw bboxes

    Args:
        image (np.ndarray): image
        bboxes (np.ndarray): detected bbox
        classes (np.ndarray): class names
        scores (np.ndarray): scores
        object_num (int): num_of_detection

    Returns:
        np.ndarray: drawed image
    """
    tmp_image = copy.deepcopy(image)

    for i in range(detect_num):
        score = round(scores[i] * 100., 2)
        bbox = bboxes[i]
        class_name = classes[i]

        y1, x1 = int(bbox[1]), int(bbox[0])
        y2, x2 = int(bbox[3]), int(bbox[2])

        cv2.putText(
            tmp_image, str(class_name) + ' - with ' +
            '{:.2f}'.format(score) + "% conficence.",
            (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            cv2.LINE_AA)
        cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return tmp_image
