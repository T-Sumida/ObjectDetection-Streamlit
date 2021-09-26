# -*- coding:utf-8 -*-
import time

from utils.ui import object_detector_ui, description, upload_image, show_image
from utils.model import detect


def run_app():
    description("Object Detector Example",
                "**TensorflowLite Model Makerで作成したObjectDetectorのサンプルモック.**")
    num_threads, model_type, confidence_threshold = object_detector_ui()

    image = upload_image()

    if image is not None:
        start_t = time.time()
        bboxes, classes, scores, detect_num = detect(
            image, num_threads, model_type, confidence_threshold)
        elapsed_time = int((time.time() - start_t) * 1000)
        show_image(image, bboxes, scores, classes, detect_num, elapsed_time)


if __name__ == "__main__":
    run_app()
