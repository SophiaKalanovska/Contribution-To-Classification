import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim

def calculate_cosine_similarity(explanation1, explanation2):
    # Flatten the explanation maps to 1D arrays
    # Calculate cosine similarity - the result is a matrix
    similarity_matrix = cosine_similarity([explanation1], [explanation2])

    # The similarity score is the first element of the matrix
    similarity_score = similarity_matrix[0, 0]
    
    return similarity_score

def calculate_euclidean_distance(explanation1, explanation2):
    # Flatten the explanation maps to 1D arrays
    explanation1_flat = explanation1.flatten()
    explanation2_flat = explanation2.flatten()

    # Calculate Euclidean distance
    distance = explanation1_flat - explanation2_flat
    print(distance)
    # squared_diffs = (explanation1_flat - explanation2_flat) ** 2
    
    return np.linalg.norm(distance)


def compare_ssim(explanation1, explanation2):
    # Flatten the explanation maps to 1D arrays
    # explanation1_flat = explanation1.flatten()
    # explanation2_flat = explanation2.flatten()

    # Calculate Euclidean distance
    # (score, diff) = compare_ssim(explanation1, explanation2)
    (score, _) = cv2.quality.QualitySSIM_create(explanation1).compute(explanation2)
    return score[0]
    
def mean_distance(v1, v2):
    """
    Calculate the mean Euclidean distance between two vectors.

    :param v1: First vector
    :param v2: Second vector
    :return: Mean Euclidean distance
    """
    # Ensure the vectors are NumPy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check if both vectors have the same length
    if v1.shape != v2.shape:
        raise ValueError("Vectors must be the same length")

    # # Calculate the Euclidean distance
    distance = np.linalg.norm(v1 - v2)
    # distance = np.sum(np.abs(v1 - v2))

    # Calculate the mean distance
    # mean_dist = distance / len(v1)
    # print(len(v1))
    return distance


def l2_normalize_both(array1, array2):
    global_min = min(np.min(array1), np.min(array2))
    global_max = max(np.max(array1), np.max(array2))

    # Apply Min-Max scaling
    scaled_array1 = (array1 - global_min) / (global_max - global_min)
    scaled_array2 = (array2 - global_min) / (global_max - global_min)

    return scaled_array1, scaled_array2

def normalise_sum_to_one(array1, array2):
    current_sum = np.sum(array1) + np.sum(array2)

    # Determine the scaling factor
    scaling_factor = 2 / current_sum

    # Scale the arrays
    scaled_array1 = array1 * scaling_factor
    scaled_array2 = array2 * scaling_factor
    # scaled_array1 = scaled_array1[scaled_array1 != 0]
    # scaled_array2 = scaled_array2[scaled_array2 != 0]

    return scaled_array1, scaled_array2 # Return the scaled arrays and their new sum



def l2_normalize(arr):
    total = sum(arr.flatten())
    if total == 0:
        return np.zeros(len(arr))
    else:
        return np.array(arr) / total


def append_results(file_path, new_results):
    # Convert new_results to DataFrame
    new_df = pd.DataFrame(new_results)

    # # Ensure new_df has the correct shape
    # if not all(isinstance(val, list) for val in new_results.values()):
    #     raise ValueError("All values in new_results must be lists of the same length.")

    # Check if file exists
    if os.path.exists(file_path):
        # Load existing data
        df = pd.read_csv(file_path)
        # Use pd.concat for appending
        updated_df = pd.concat([df, new_df], ignore_index=True)
    else:
        # If file doesn't exist, new_df is the updated DataFrame
        updated_df = new_df

    # Save to CSV
    updated_df.to_csv(file_path, index=False)



