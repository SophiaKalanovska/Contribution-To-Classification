import numpy as np
import cv2

def inverted_masks(x, masks = []):
    # size_2 = x.shape[1], x.shape[2], 3
    # proba_0 = 0.99982
    # random_mask_2 = np.expand_dims(np.random.choice([0.0, 1.0], size=size_2, p=[proba_0, 1 - proba_0]), 0)

    # zero_mask = np.zeros_like(masks[0])
    # one_mask = np.ones_like(masks[0])
    # all_regions = np.sum(masks, 0, keepdims=True)[0]
    # inverted_masks = [~mask for mask in masks]
    summed_mask = np.sum(masks, 0, keepdims=True)[0]
    inverted_summed_mask = summed_mask == 0

    mask_uint8 = (inverted_summed_mask * 255).astype(np.uint8)

    # Apply connected components to the binary mask
    num_labels, labels = cv2.connectedComponents(mask_uint8)

    min_area_threshold = 0.001 * x.shape[1]*  x.shape[2]
    # invert = np.sum(inverted_masks, 0, keepdims=True)[0]
    # all_regions = np.sum(masks, 0, keepdims=True)[0]
    # masks.append(zero_mask)
    # masks.append(random_mask_2)

    for label in range(1, num_labels):  # Start from 1 to ignore the background
        component_mask = (labels == label).astype(np.uint8) * 1
        area = np.sum(component_mask)
        if area >= min_area_threshold:
            masks.append(component_mask.astype(np.bool_))

    # masks.append(inverted_summed_mask)
    # masks.append(one_mask)
    # return [all_regions, one_mask]
    return masks