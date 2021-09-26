# -*- coding:utf-8 -*-
from typing import Optional, Tuple, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.model import MODEL_TYPE, draw_bboxes


def description(header: str, description: str):
    """show description

    Args:
        header (str): header message
        description (str): description text
    """
    st.subheader(header)
    st.markdown(description)


def object_detector_ui() -> Tuple[int, str, float]:
    """show object detector ui in sidebar

    Returns:
        Tuple[int, str, float]: [number of threads, model type string, threshold]
    """
    st.sidebar.markdown("# Model Config")
    num_thread = st.sidebar.slider("Number of Thread", 1, 4, 1, 1)
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    model_type = st.sidebar.radio("Model Type", MODEL_TYPE)

    return num_thread, model_type, confidence_threshold


def upload_image() -> Optional[np.ndarray]:
    """show upload image area

    Returns:
        Optional[np.ndarray]: uploaded image
    """
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "JPG"])
    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        return image
    else:
        return None


def show_image(image: np.ndarray, bboxes: List, scores: List, classes: List, detect_num: int, elapsed_time: int):
    """show processed image.

    Args:
        image (np.ndarray): original image
        bboxes (List): detected bounding box
        scores (List): detected score
        classes (List): detected class names
        detect_num (int): number of detection
        elapsed_time (int): processing time
    """
    image = draw_bboxes(image, bboxes, scores, classes, detect_num)
    image = cv2pil(image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.markdown("**elapsed time : " + str(elapsed_time) + "[msec]**")
    pass


def cv2pil(image: np.ndarray) -> Image:
    """cv2 image to PIL image

    Args:
        image (np.ndarray): cv2 image

    Returns:
        Image: PIL image
    """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
