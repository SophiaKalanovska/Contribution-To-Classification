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
import json


###############################################################################
###############################################################################

base_dir = os.path.dirname(__file__)
utils = SourceFileLoader("utils", os.path.join(base_dir, "utils.py")).load_module()

###############################################################################
###############################################################################

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    print("I am in this file")
    image_path = sys.argv[1]
    # image_path = "ILSVRC2012_val_00000068_big_uniform.JPEG"
    print(image_path)
    
    
    image_folder = sys.argv[2]
    noise_type = sys.argv[3]

    # image_folder = "ILSVRC_big" 
    # noise_type = "_big_uniform"


    with open("/root/REVEAL_SAM/examples/temp_results.json", 'r') as temp_file:
        list_of_returns = json.load(temp_file)
        

    # print(list_of_returns)
    # list_of_returns = json.loads(list_of_returns)
    # print(list_of_returns)



    sorted_mask = json.loads(list_of_returns[0])
    sorted_mask = np.array(sorted_mask)
    sorted_mask = [np.array(element) for element in sorted_mask]

    sorted_masks_3D = json.loads(list_of_returns[1])
    sorted_masks_3D = np.array(sorted_masks_3D)
    sorted_masks_3D = [np.array(element) for element in sorted_masks_3D]

    norm_lrp = json.loads(list_of_returns[2])
    norm_lrp = np.array(norm_lrp)

    norm_reveal = json.loads(list_of_returns[3])
    norm_reveal = np.array(norm_reveal)

    the_label_index = json.loads(list_of_returns[4])
    the_label_index = np.array(the_label_index)

    predictions = list_of_returns[5]
    print(the_label_index)

    

    # image_path = "ILSVRC2012_val_00000001_gausian_big.JPEG"
    print(image_path)

    image_size = 224
    # image_size = 299
    image = utils.load_image(
        os.path.join(base_dir, image_folder, image_path), image_size)
    image_new = image[:, :, :3]

    image_bgr = np.asarray(image_new, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)



    # CHECKPOINT_PATH = "/root/REVEAL_SAM/sam_vit_h_4b8939.pth"
    # # CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # MODEL_TYPE = "vit_h"

    # sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    # sam.to(device=DEVICE)

    # mask_generator = SamAutomaticMaskGenerator(sam)
    # IMAGE_PATH = base_dir + "/images/"+ image_path

    # image_bgr = np.asarray(image_new, dtype=np.uint8)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # result = mask_generator.generate(image_rgb)



    # mask_annotator = sv.MaskAnnotator()
    # detections = sv.Detections.from_sam(result)


    model_call = vgg16
    # model_call = vgg19
    # model_call = inception
    # model_call = res

    model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # model, preprocess = res.ResNet50(), res.preprocess_input
    # model, preprocess = vgg19.VGG19(), vgg19.preprocess_input
    # model, preprocess = inception.InceptionV3(), inception.preprocess_input

    # Strip softmax layerexamples

    model = innvestigate.model_wo_softmax(model)
    x = preprocess(image[None])

    # masks, masks_from_heatmap3D = innvestigate.masks_from_heatmap.retrieve(result, x, image_size)

    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model)
    pr_2 = model.predict_on_batch(x)
    the_label_index_2 = np.argmax(pr_2, axis=1)

    print(the_label_index_2)

    # # distribute the relevance to the input layer
    start_time = time.time()
    a = analyzer.analyze(x, the_label_index)

    # norm_lrp_2 = innvestigate.faithfulnessCheck.calculate_distance.l2_normalize(a)


    # masks_pixels, masks_from_heatmap3D_pixels = innvestigate.masks_from_heatmap.retrieve_pixels(a, x, image_size, image_path)

    # sorted_mask, sorted_masks_3D = innvestigate.masks_from_heatmap.rank_and_sort_masks(masks, masks_from_heatmap3D, masks_from_heatmap3D_pixels)

    # if len(sorted_mask)> 9:
    #     sorted_mask, sorted_masks_3D = sorted_mask[:9], sorted_masks_3D[:9]
    # sorted_mask = [sorted_mask[i, ...] for i in range(len(sorted_mask))]
    # sorted_masks_3D = [sorted_masks_3D[i, ...] for i in range(len(sorted_mask))]
    # sorted_mask.append(np.ones_like(sorted_mask[0]))
    # sorted_masks_3D.append(np.ones_like(sorted_masks_3D[0]))

    analyzer = innvestigate.create_analyzer("reveal.alpha_2_beta_1", model, **{"masks": sorted_masks_3D, "index": the_label_index})


    # # # Apply analyzer w.r.t. maximum activated output-neuron
    start_time = time.time()
    relevance = analyzer.analyze(x, label=the_label_index)
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    masks_times_relevance = sorted_masks_3D * relevance[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    exp = masks_times_relevance[:-1].sum(axis=0)
    # norm_reveal_2 = innvestigate.faithfulnessCheck.calculate_distance.l2_normalize(exp)



    # illustrate = innvestigate.illustrate_clusters.Illustrate()
    # illustrate.mask_to_input_relevance_of_mask(relevance, sorted_masks_3D, scene_colour = copy.copy(image_rgb), masks = sorted_mask, image_path = image_path, label=predictions)
        # illustrate.mask_to_input_relevance_of_pixels([random.randint(0, 100) for _ in range(len(masks_pixels)+2)], masks_from_heatmap3D_pixels, label = predictions[0][0][1], image_name= image_path)
    
    
    
    norm_reveal, norm_reveal_2 = innvestigate.faithfulnessCheck.calculate_distance.normalise_sum_to_one(norm_reveal.flatten(), exp.flatten())

    # Reveal_cosine = innvestigate.faithfulnessCheck.calculate_distance.calculate_cosine_similarity(norm_reveal, norm_reveal_2)
    Reveal_euclidean = innvestigate.faithfulnessCheck.calculate_distance.mean_distance(norm_reveal, norm_reveal_2)
    # Reveal_euclidean = innvestigate.faithfulnessCheck.calculate_distance.compare_ssim()

    norm_lrp, norm_lrp_2 = innvestigate.faithfulnessCheck.calculate_distance.normalise_sum_to_one(norm_lrp.flatten(), a.flatten())   
    # LRP_cosine = innvestigate.faithfulnessCheck.calculate_distance.calculate_cosine_similarity(norm_lrp, norm_lrp_2)
    LRP_euclidean = innvestigate.faithfulnessCheck.calculate_distance.mean_distance(norm_lrp, norm_lrp_2)
    # LRP_euclidean = innvestigate.faithfulnessCheck.calculate_distance.compare_ssim(norm_lrp.flatten(), a.flatten())

    change = the_label_index_2 == the_label_index
    print(change)


    name, extension = image_path.rsplit('.', 1)
   
    results = {
    #  f"REVEAL_{noise_type}_cosine": Reveal_cosine,
     f"REVEAL_{noise_type}_euclidean": Reveal_euclidean,
    #  f"LRP_{noise_type}_cosine": LRP_cosine,
     f"LRP_{noise_type}_euclidean": LRP_euclidean,
     f"Same_Classification_{noise_type}": change,
    }

    innvestigate.faithfulnessCheck.calculate_distance.append_results('results_vgg16.csv', results)


    # innvestigate.faithfulnessCheck.calculate_distance.append_results('input_invaraince_explanation_method_comparison_eucliden.csv', results_euc)
         