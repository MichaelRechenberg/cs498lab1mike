# Main script that continuously captures images from
#   the Pi's camera and TODO: runs object recognition
#   on it using Tensorflow, and persists results to 
#   permanent storage
import logging

from picamera import PiCamera
from picam_wrapper import PiCameraWrapper


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
camera.resolution = (1280, 720)

# See PiCameraWrapper.capture_continuous() for what "array resolution" is
array_resolution = (1280, 720)



# TODO: have this callback actually perform the CV
def dummy_callback(pic_arr):
  # mimic some processing
  print(pic_arr.shape)
  import time
  time.sleep(2)
  print(pic_arr)
  time.sleep(5)

callback = dummy_callback
  


# Start continuous capture (ends when process is killed/interrupted)
# If we want finer control over capture, we could use PiCameraWrapper.capture()
logger.info("Starting continous camera capture")
PiCameraWrapper.capture_continuous(
    camera,
    callback, 
    array_resolution,
    format,
    logger
)

logger.info("Stopped continous camera capture")


