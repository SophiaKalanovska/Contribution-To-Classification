from jenkspy import JenksNaturalBreaks
import numpy as np
import jenkspy
import seaborn
import matplotlib.pyplot as plt


# def jenks_breaks(activations, numberOfBreaks, image_size, n, cutoff, x):
#     result = activations.flatten()
#     image = x
#     top_nity = np.percentile(result, 95)

#     activation_ranges = []

#     y = -1
#     for i in activations:
#         y += 1
#         x = -1
#         for j in i:
#             x += 1
#             for r in range(n):
#                 if abs(j) > top_nity:
#                     activation_ranges.append((x , image_size - y, image[0][y][x][0] / 3, image[0][y][x][1] / 3, image[0][y][x][2]/ 3))
#                     continue

#     return activation_ranges

def jenks_breaks(activations, numberOfBreaks, image_size, n, cutoff, x):
    result = activations.flatten()
    
    image = x
    # quaters = np.floor(result.size / n)
    top_nity = np.percentile(result, 95)
    innerbreaks = []
    jnb = JenksNaturalBreaks(numberOfBreaks)
    n, m = n, n  # Define the number of rows and columns for the grid. You can adjust these as needed.

    square_height = np.int32(np.ceil(activations.shape[0] / n))
    square_width = np.int32(np.ceil(activations.shape[1] / m))
    for row in range(n):
        for col in range(m):
            square_activations = activations[row * square_height: (row + 1) * square_height, 
                                            col * square_width: (col + 1) * square_width]
            square_flattened = square_activations.flatten()
            
            jnb.fit(square_flattened)
            innerbreaks.append(jnb.breaks_)

    activation_ranges = []


    for y in range(activations.shape[0]):
        for x in range(activations.shape[1]):
            # Determine which square the current activation belongs to
            square_row = y // square_height
            square_col = x // square_width
            square_idx = m  * square_row + square_col 
            j = activations[y, x]
            if abs(j) > innerbreaks[square_idx][-5] and abs(j) > top_nity:
            # if abs(j) > innerbreaks[square_idx][-5]:
                activation_ranges.append((x, image_size - y, j*image_size*2, image[0][y][x][0]/3, image[0][y][x][1]/3, image[0][y][x][2]/3))        
                
    return activation_ranges

#
# def remove_bottom(activations, numberOfBreaks, image_size):
#     result = np.array(activations).flatten()
#     jnb = JenksNaturalBreaks(numberOfBreaks)
#     jnb.fit(result)
#     innerbreaks = jnb.breaks_
#
#     activation_ranges = []
#
#
#     y = -1
#     for i in activations:
#         y += 1
#         x = -1
#         for j in i:
#             x += 1
#             if j > innerbreaks[1]:
#                 activation_ranges.append((x, image_size - y))
#                 continue
#     #
#     if len(activation_ranges) > 2500:
#         activation_ranges = remove_bottom(activations, numberOfBreaks - 2, image_size)
#     return activation_ranges

def print_jenks_data(jnb):
    try:
        print(jnb.labels_)
        print(jnb.groups_)
        print(np.len(jnb.groups_[0]))
        print(jnb.inner_breaks_)
        print(jnb.breaks_)
    except:
        pass


def visualise_breaks(activations):

    result = activations.flatten()
    breaks = jenkspy.jenks_breaks(result, nb_class=5)
    print(breaks)

    plt.figure()
    seaborn.stripplot(x=result, jitter=True)
    seaborn.despine()
    locs, labels = plt.xticks()
    plt.xticks(locs, map(lambda x: x, np.round(locs,2)))
    plt.xlabel('Intensity')
    plt.yticks([])
    for b in breaks:
        plt.vlines(b, ymin=-0.2, ymax=0.5)

    plt.show()
