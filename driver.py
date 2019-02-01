# Main script to continuously capture images from
#  the Pi's camera, run object recognition on it, 
#  and communicate with brake sensor and head unit
import logging
import os
import math

from picamera import PiCamera
import cv2

from picam_wrapper import PiCameraWrapper
from object_recognition import ObjectRecognition


# ================= INITIALIZATION ==========================

# Init logging
LOGGER_NAME = "LAB1_LOGGER"
LOG_FILE = "/tmp/lab1.log"

logger = logging.getLogger(LOGGER_NAME)

fh = logging.FileHandler(LOG_FILE)
log_formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(log_formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


# Init Raspberry Pi's camera
# https://www.raspberrypi.org/documentation/hardware/camera/ for resolution/fps tradeoffs
# PiCamera crashed with 1080p
logger.info("Initializing camera")
format = 'rgb'
camera = PiCamera()
CAMERA_HORIZONTAL_RESOLUTION = 720
CAMERA_VERTICAL_RESOLUTION = 1280
camera.resolution = (CAMERA_VERTICAL_RESOLUTION, CAMERA_HORIZONTAL_RESOLUTION)

# See PiCameraWrapper.capture_continuous() for what "array resolution" is
CAPTURE_ARRAY_RESOLUTION = (CAMERA_VERTICAL_RESOLUTION, CAMERA_HORIZONTAL_RESOLUTION)

# Init object recognition
logger.info("Initializing ObjectRecognition object")
# TODO: change getcwd() to env variable if we want
MODEL_ROOT_DIR = os.path.join(os.getcwd(), "models")
# Not using a quantized model at the moment since the quantized models 
#   now use Tensorflow Lite and there isn't a pretty binary to use 
#   for Tensorflow Lite on Pi (See Discord for more discussion)
MODEL_NAME = "ssdlite_mobilenet_v2_coco_2018_05_09"

# The minimum threshold to use for *detecting* objects. Note that this
#  is not necessarily the threshold used for determining to send
#  the brake signal or not
SCORE_THRESHOLD = 0.2
object_recognition_model = ObjectRecognition(MODEL_ROOT_DIR, MODEL_NAME)


# ================= INITIALIZATION ==========================




# =================== CORE LOOP =============================

# Trying to perform inference on a high resolution image will take too long
#  and the models were trained on lower-resolution images like (300x300) or 
#  (600x600). So we will first resize the raw image captured from the 
#  camera to these dimensions before passing the resized image to the 
#  module that performs object recognition
RESIZED_HORIZONTAL_RESOLUTION = 300
RESIZED_VERTICAL_RESOLUTION = 300

# Not using PiCameraWrapper.capture_continuous() for demonstration purposes
#  and we may need finer control for multi-threading/networking purposes
logger.info("Starting continous camera capture")
while True:

    logger.info("Capturing image")
    raw_capture_arr = PiCameraWrapper.capture(
        camera,
        CAPTURE_ARRAY_RESOLUTION,
        format
    )

   

    resized_capture_arr = cv2.resize(
        raw_capture_arr,
        (RESIZED_HORIZONTAL_RESOLUTION, RESIZED_VERTICAL_RESOLUTION),
        interpolation=cv2.INTER_LINEAR
    )

    logger.info("Performing inference")

    annotated_image, valid_classes, valid_scores = object_recognition_model.detect_objects(
            resized_capture_arr,
            SCORE_THRESHOLD
    )

    logger.info("We performed inference")

    # TODO: Send annotated image and classes/scores to head unit
    # TODO: Send brake light signal to brake unit, receive image data and make decision based on that


logger.info("Stopped continous camera capture")
camera.close()


