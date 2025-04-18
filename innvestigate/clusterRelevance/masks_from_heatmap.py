import numpy as np
from . import  one_dimensional_breaks, two_dimensional_clusters, random_clusters, masks_from_heatmap
import matplotlib.pyplot as plt
import cv2

def remove_contained_parts(bigger_mask, smaller_masks):
    # Ensure the bigger_mask is a numpy array and of dtype np.uint8
    bigger_mask = np.array(bigger_mask)
    bigger_mask = bigger_mask.astype(np.uint8)

    # Iterate through each smaller mask and remove its part from the bigger mask
    for smaller_mask in smaller_masks:
        # Ensure the smaller_mask is a numpy array and of dtype np.uint8
        smaller_mask = np.array(smaller_mask)
        smaller_mask = smaller_mask.astype(np.uint8)

        # Check if the smaller mask is contained in the bigger mask
        # if is_mask_contained(bigger_mask, smaller_mask):
            # Remove the part of the bigger mask that is contained by the smaller mask
            
        bigger_mask = cv2.bitwise_xor(bigger_mask, cv2.bitwise_and(bigger_mask, smaller_mask))

    return bigger_mask

def count_ones(mask):
    return np.count_nonzero(mask == 1)

def calculate_overlap_and_coverage(mask1, mask2):
    # Calculate overlap
    overlap = np.logical_and(mask1, mask2)
    overlap_count = np.sum(overlap)

    # Calculate total '1's in each mask
    total_ones_mask1 = np.sum(mask1)
    total_ones_mask2 = np.sum(mask2)

    # # Calculate coverage percentages
    coverage_mask1 = (overlap_count / total_ones_mask1) if total_ones_mask1 > 0 else 0
    coverage_mask2 = (overlap_count / total_ones_mask2) if total_ones_mask2 > 0 else 0

    # # Combine metrics (example: average coverage)
    combined_metric = (coverage_mask1 + coverage_mask2) / 2

    return combined_metric


def rank_and_sort_masks(masks0, masks1, masks2):
    masks0 = np.array(masks0)
    masks1 = np.array(masks1)
    masks2 = np.array(masks2)
    overlaps = []

    for mask1 in masks1:
        overlap_sum = 0
        for mask2 in masks2:
            overlap = np.sum(np.logical_and(mask1, mask2))
            overlap_sum += overlap
        overlaps.append(overlap_sum)

    ranked_indices = np.argsort(overlaps)[::-1]
    # # relevances_sorted, masks_with_ones_sorted = zip(*sorted(zip(relevances_clusters, masks), key = lambda x:x[0], reverse=True))
    sorted_mask = masks0[ranked_indices, ...]
    sorted_masks_3D = masks1[ranked_indices, ...]
    #     masks0 = np.array(masks0)
    # masks1 = np.array(masks1)
    # masks2 = np.array(masks2)

    overlaps_percent = []

    for mask1 in masks1:
        overlap_sum = 0
        # total_ones_in_mask1 = np.sum(mask1)  # Sum of all 1s in mask1
        covarage = []
        for mask2 in masks2:
            # total_ones_in_mask2 = np.sum(mask2)
            overlap = calculate_overlap_and_coverage(mask1, mask2)
            # overlap = np.sum(np.logical_and(mask1, mask2))
            covarage.append(overlap)
        # if total_ones_in_mask1 != 0:  # Avoid division by zero
        #     overlap_percent = (overlap_sum / total_ones_in_mask2) * 100  # calculate percentage
        # else:
        #     overlap_percent = 0
        # overlaps_percent.append(overlap)
        overlaps_percent.append(np.max(covarage))
    # sorted_mask, sorted_masks_3D, overlaps_percent_2 = zip(*sorted(zip(masks0, masks1, overlaps_percent), key = lambda x:x[2], reverse=True))
    
    
    ranked_indices = np.argsort(overlaps_percent)[::-1]
    sorted_mask = masks0[ranked_indices, ...]
    sorted_masks_3D = masks1[ranked_indices, ...]
   
    return sorted_mask, sorted_masks_3D


def expand_cluster(mask, num_pixels=5):
    # Define the kernel for dilation (a square kernel with side length of num_pixels)
    kernel = np.ones((num_pixels, num_pixels), np.uint8)

    # Perform dilation on the mask
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    return dilated_mask


def retrieve_pixels(a, x, size, image_name):
    # b = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    # b /= np.max(np.abs(b))
    activations = a[0]
    numberOfBreaks = 6
    cutoff = 1000
    n =4

    categories = one_dimensional_breaks.jenks_breaks(activations, numberOfBreaks, activations[-1].size, n, cutoff, x)

    # Visualise the activation ranges
    # one_dimensional_breaks.visualise_breaks(b[0])


    # Extracting x and y coordinates
    x_coords = [vector[0] for vector in categories]
    y_coords = [vector[1] for vector in categories]

    # Plotting
    # plt.scatter(x_coords, y_coords, color='red', s=1)  # 's=1' sets the size of markers to 1 for pixel-like appearance
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(False)
    # plt.xlim(0, size)
    # plt.ylim(0, size)
    # plt.savefig(image_name + "_heatmap.png")
    # plt.show()
    # plt.clf()

    # Create masks for each cluster found on the level with highest activation
    masks, masks_with_ones = two_dimensional_clusters.DbSCAN_for_activations(categories, activations[-1].size, a)

    # masks_3D = random_clusters.random_clusters(b, masks_with_ones)

    return masks, masks_with_ones

def retrieve(a, x, size):
    masks = [
        mask['segmentation']
        for mask
        in sorted(a, key=lambda x: x['area'], reverse=True)
    ]
    

    num_masks = len(masks)
    # containment_matrix = np.zeros((num_masks, num_masks), dtype=bool)
    masks_two_channel = []
    for i in range(num_masks):
        num_pixels = x.shape[1]/ 100
        masks[i] = expand_cluster(masks[i], np.ceil(num_pixels).astype(np.int32))

    # for i in range(num_masks):
    #     masks[i] = remove_contained_parts(masks[i], masks[i+1:])
    #     num_pixels = x.shape[1]/ 100
    #     masks[i] = expand_cluster(masks[i], np.ceil(num_pixels).astype(np.int32))

    # masks = sorted(masks, key=count_ones, reverse=True)[: 6]
    num_masks = len(masks)
    for i in range(num_masks):
        masks[i] = remove_contained_parts(masks[i], masks[i+1:])     
        masks_two_channel.append(masks[i].astype(np.bool_))


    masks_two_channel = random_clusters.inverted_masks(x, masks_two_channel)

    masks_resized = []
    for ret in masks_two_channel:
        ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
        if ret.ndim == 2:
            ret = np.expand_dims(ret, -1)
            masks_resized.append(np.expand_dims(np.repeat(ret, 3, axis=-1), 0))    


    
    masks_resized, masks_two_channel = zip(*sorted(zip(masks_resized, masks_two_channel), key = lambda x: count_ones(x[0]), reverse=True))
    masks_resized = list(masks_resized)
    masks_two_channel = list(masks_two_channel)

    return masks_two_channel, masks_resized