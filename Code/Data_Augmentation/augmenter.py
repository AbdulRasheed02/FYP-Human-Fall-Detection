import cv2
import h5py
import os
import sys
import numpy as np

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from parameters import (
    project_directory,
    dataset_directory,
    background_subtraction_algorithms,
    background_subtraction_algorithm,
)

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Basically accepts a list of images and then augments that list of images
def augment_images(image_list):
  angle_range = (-10, 10)  # Rotate images between -10 and 10 degrees
  brightness_range = (0.8, 1.2)  # Adjust brightness between -20% and 20%
  """
  Augments an image with random rotations, scaling, and brightness adjustments.

  Args:
      image: The image to be augmented (numpy array).
      angle_range: Tuple representing the range of random rotations (in degrees).
      scale_range: Tuple representing the range of random scaling factors.
      brightness_range: Tuple representing the range of random brightness adjustments.

  Returns:
      A list of augmented images.
  """
  augmented_images = []
  for image in image_list:
    rows, cols, ch = image.shape
    # Random rotation
    angle = np.random.uniform(angle_range[0], angle_range[1])
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    final_img = cv2.warpAffine(image, M, (cols, rows))
    # Random brightness adjustment
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    final_img = cv2.convertScaleAbs(final_img, alpha=brightness, beta=0)
    # horizontal flips 
    final_img = cv2.flip(final_img, 1)
    final_img = np.expand_dims(final_img, axis=-1)
    augmented_images.append(final_img)
  augmented_images = np.array(augmented_images)
  return augmented_images



# For using preprocessed images from h5py as input
name = "Thermal_T3"
path = "{}\Dataset\H5PY\Data_set-{}-imgdim64x64.h5".format(project_directory, name)
with h5py.File(path, "r") as hf:
    data_dict = hf["{}/Processed/Split_by_video".format(name)]
    # Any fall or ADL di3rectory
    vid_total = data_dict["Fall0"]["Data"][:]
    augmented_images = augment_images(vid_total)
    print(augmented_images[0].shape)
