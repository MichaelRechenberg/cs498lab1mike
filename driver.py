# Main script that continuously captures images from
#   the Pi's camera and TODO: runs object recognition
#   on it using Tensorflow, and persists results to 
#   permanent storage
import logging

from picamera import PiCamera
from picam_wrapper import PiCameraWrapper


# Init logging
LOGGER_NAME = "LAB1_LOGGER"

logfile = "/tmp/lab1.jog"
logging.basicConfig(
   filename=logfile,
   filemode='w',
   format='%(levelname) - %(asctime)s - %(message)s'
)
logger = logging.getLogger(LOGGER_NAME)


# Init Raspberry Pi's camera
camera = PiCamera()

# TODO: have this callback actually perform the CV
callback = (lambda filepath: print("Yep, that's a filepath {0}".format(filepath)))


# Start continuous capture (ends when process is killed/interrupted)
logging.info("Starting continous camera capture")
PiCameraWrapper.capture_continuous(
    camera,
    callback, 
    LOGGER_NAME
)

logging.info("Stopped continous camera capture")


