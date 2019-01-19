import logging
import os

class PiCameraWrapper:
  def capture_continuous(camera, post_capture_callback, logger_name, tmp_dir="/tmp"):
    """Capture images from this PiCamera as fast as possible,
        invoking a specified callback after an image has been
        captured by the camera.

       Note that after the callback is called, this method will
        delete the image...so consumers may want to save a copy
        of the captured image to another directory if they
        want to persist the images over time

       Arguments:
         camera - The PiCamera to perform continuous capture with
         post_capture_callback - A function that takes an absolute filepath
           to an image and performs some processing with it. After this
           callback returns, the image at the filepath given as the 
           callback's argument will be deleted by this method
         logger_name - The name of the logger to use for logging (we assume
             that the logger has been configured after this method is called)
         tmp_dir (optional) - Absolute filepath to an existing directory
           that this function uses to temporarily store captured images
    """

    logger = logging.getLogger(logger_name)

    # TODO: maybe we could pass around open file-descriptor rather than
    #   save to disk
    image_base_format_str = "image{counter}.jpg"
    output_filepath = os.path.join(tmp_dir, image_base_format_str)
    for most_recent_img_filepath in camera.capture_continuous(output_filepath):
      logger.info("Most recently captured image saved to {0}".format(most_recent_img_filepath))

      logger.info("Invoking callback")
      post_capture_callback(most_recent_img_filepath)
      logger.info("Callback completed, now removing image {0}".format(most_recent_img_filepath))

      # TODO: uncomment this line after testing
      # os.remove(most_recent_img_filepath)





    

