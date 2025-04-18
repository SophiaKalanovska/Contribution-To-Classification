# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:

from __future__ import \
    absolute_import, print_function, division, unicode_literals

import numpy as np
np.random.seed(42)
# import innvestigate as inn
from importlib.machinery import SourceFileLoader
import copy
import random
# from clusterRelevance import illustrate_clusters
# from clusterRelevance import masks_from_heatmap
# from sklearn.preprocessing import StandardScaler
# from tensorflow.python.framework.opscd .. import enable_eager_execution_internal
import os
import sys
import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv
import pandas as pd
import numpy as np
from PIL import Image



file_dir = os.path.dirname(__file__)

sys.path.append(file_dir)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import innvestigate
import innvestigate.utils
# from innvestigate.utils import *
import tensorflow.keras.applications.inception_v3 as inception
import keras.applications.vgg16 as vgg16
import keras.applications.resnet as res
import keras.applications.vgg19 as vgg19

import keras.applications.inception_v3 as inception

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
# import argparse
# import pickle


###############################################################################
###############################################################################

base_dir = os.path.dirname(__file__)
utils = SourceFileLoader("utils", os.path.join(base_dir, "utils.py")).load_module()

###############################################################################
###############################################################################

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    if len(sys.argv) < 1:
        print("Usage: python my_program.py <image_path>")
        sys.exit(1)

    # image_path = sys.argv[1]
    # image_path = "ILSVRC2012_val_00000001.JPEG"
    
    # image_size = 224
    # # image_size = 299
    # image = utils.load_image(
    #     os.path.join(base_dir, "ILSVRC", image_path), image_size)
    # image_new = image[:, :, :3]

# 
    image_new = cv2.imread("/Users/sophia/Documents/EVERYTHING_PHD/REVEAL_SAM/examples/images/grasshoper.JPEG")
    image_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)

    name, extension = os.path.splitext("grasshoper.jpeg")


    image_blur =innvestigate.faithfulnessCheck.noise_and_blur.gaussian_blur(image_new)
    # image_blur = image_blur.astype(np.uint8)
    image_blur = Image.fromarray(image_blur)
    image_blur.save("/Users/sophia/Documents/EVERYTHING_PHD/REVEAL_SAM/examples/ILSVRC_blur/" + name + "blur" + extension)



    image_gausian_small = innvestigate.faithfulnessCheck.noise_and_blur.add_gaussian_noise(image_new, 0.1)
    # image_gausian_small = image_gausian_small.astype(np.uint8)
    image_gausian_small = Image.fromarray(image_gausian_small)
    image_gausian_small.save("/Users/sophia/Documents/EVERYTHING_PHD/REVEAL_SAM/examples/ILSVRC_small/" + name + "_gausian_small_normal" + extension)

    image_gausian_big = innvestigate.faithfulnessCheck.noise_and_blur.add_gaussian_noise(image_new, 50.0)
    # image_gausian_big = image_gausian_small.astype(np.uint8)
    image_gausian_big = Image.fromarray(image_gausian_big)
    image_gausian_big.save("/Users/sophia/Documents/EVERYTHING_PHD/REVEAL_SAM/examples/ILSVRC_big_normal/" + name + "_gausian_big_normal" + extension)

    image_uniform_noise = innvestigate.faithfulnessCheck.noise_and_blur.add_uniform_noise(image_new)
    # image_gausian_big = image_gausian_big.astype(np.uint8)
    image_uniform_noise = Image.fromarray(image_uniform_noise)
    image_uniform_noise.save("/Users/sophia/Documents/EVERYTHING_PHD/REVEAL_SAM/examples/ILSVRC_big/" + name + "_big_uniform" + extension)

    image_uniform_noise_small = innvestigate.faithfulnessCheck.noise_and_blur.add_uniform_noise(image_new, 0.02)
    # image_gausian_big = image_gausian_big.astype(np.uint8)
    image_uniform_noise_small = Image.fromarray(image_uniform_noise_small)
    image_uniform_noise_small.save("/Users/sophia/Documents/EVERYTHING_PHD/REVEAL_SAM/examples/ILSVRC_small_uniform/" + name + "_small_uniform" + extension)
