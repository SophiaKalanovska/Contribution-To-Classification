import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
from typing import List, Optional, Union
from supervision.draw.color import Color, ColorPalette
import copy
import skimage.color as skc
# import sk.color.gray2rgb
import supervision as sv
from random import randrange
import numpy as np
np.random.seed(43)
import cv2


class Illustrate:
    """
    A class for overlaying masks on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to fill the mask, can be a single color or a color palette
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
    ):
        self.color: Union[Color, ColorPalette] = color



        self.primes = np.array([ 3,   5,  11,  13,  17,  19,  23,  29,  31,  37, 41, 43])
    


    # self.color: Union[Color, ColorPalette] = color
        # , 
        #     41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,
        #    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
        #    167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        #    239, 241, 251, 257, 263, 269, 271, 277

        self.color_map = {0: np.array([255, 255, 255]), # white
                        
                    # 3: np.array([randrange(255),randrange(255),randrange(255)]), #dark red 
                    # 5: np.array([randrange(255),randrange(255),randrange(255)]), # red
                    # #  7: np.array([255, 102, 102]), # light red
                    # 11: np.array([randrange(255),randrange(255),randrange(255)]),# orange
                    # 13: np.array([randrange(255),randrange(255),randrange(255)]),# light orange
                    # 17: np.array([randrange(255),randrange(255),randrange(255)]),# yellow
                    # 19: np.array([randrange(255),randrange(255),randrange(255)]), # green
                    # 23: np.array([randrange(255),randrange(255),randrange(255)]), # blueish green
                    # 29: np.array([randrange(255),randrange(255),randrange(255)]), # light blue
                    # 31: np.array([randrange(255),randrange(255),randrange(255)]),  # blue
                    # 37: np.array([randrange(255),randrange(255),randrange(255)]), # dark blue
                    # 41: np.array([randrange(255),randrange(255),randrange(255)]), # purple
                    # 43: np.array([randrange(255),randrange(255),randrange(255)]), #black
                    # 53: np.array([randrange(255),randrange(255),randrange(255)]),
                    # 59: np.array([randrange(255),randrange(255),randrange(255)]),
                    # 61: np.array([randrange(255),randrange(255),randrange(255)]),
                    # 67: np.array([randrange(255),randrange(255),randrange(255)]),
                    # 71: np.array([randrange(255),randrange(255),randrange(255)]),
                    # 73: np.array([randrange(255),randrange(255),randrange(255)]),

                    3: np.array([255, 0, 0]), # red 
                    5: np.array([253, 44, 129]), # pink
                    11: np.array([251, 118, 11]),# orange
                    13: np.array([255, 255, 0]),# yellow
                    17: np.array([90,204,0]), # light green
                    19: np.array([0,160,52]), # green
                    23: np.array([137,207,240]), # blueish green
                    29: np.array([0, 128, 255]), # light blue
                    31: np.array([0, 0, 204]),  # blue
                    37: np.array([0, 0, 102]), # dark blue
                    41: np.array([51, 0, 102]), # purple
                    
                    
                # 41: np.array([255, 180, 80]),
                # 43: np.array([255, 180, 10]),
                # -999: np.array([255, 180, 255]),
                } # light red
    def rgb2gray(self, rgb):

        grey = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return skc.gray2rgb(grey)
    
    def mask_to_input_relevance_of_mask(self, relevance, masks_from_heatmap3D, scene_colour, masks, image_path, label = None):
        opacity = 0.6
        print(relevance)
        full_relevance = relevance[-1]
        # regions_relevance = relevance[-2]
        # relevances_clusters = relevance[:-2]
        relevances_clusters = relevance[:-1]
        # masks = masks_from_heatmap3D[:-2]

        # hey = []
        # whole = masks[0]
        # for m in masks: 
        #     hey.append(m.astype(int) * b)
        #     whole = np.logical_or(m, whole)

        # last = np.invert(whole).astype(int)
        # masks.append(last)
        # hey.append(last * b)

        # result = []
        # for e in hey:
        #     n = np.count_nonzero(e)
        #     sum = np.sum(e)
        #     result.append(sum/n)

        # relevances_clusters = result 
        image = copy.copy(scene_colour)
        scene = self.rgb2gray(copy.copy(scene_colour))
        

        
        
        relevances_sorted, masks_with_ones_sorted = zip(*sorted(zip(relevances_clusters, masks), key = lambda x:x[0], reverse=True))


        custom_lines = []

        for i in range(len(masks_with_ones_sorted)):
                if i >= 10:
                    relevance = np.array([randrange(255),randrange(255),randrange(255)])
                    color = Color(relevance[0], relevance[1], relevance[2])
                else:
                    relevance = self.primes[i]
                    color = Color(self.color_map[relevance][0], self.color_map[relevance][1], self.color_map[relevance][2])

                mask = masks_with_ones_sorted[i]
                colored_mask = np.zeros_like(scene, dtype=np.uint8)
                colored_mask[:] = color.as_bgr()

                scene = np.where(
                    np.expand_dims(mask, axis=-1),
                    np.uint8(opacity * colored_mask + (1 - opacity) * scene),
                    scene,
                )

                custom_lines.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                    markerfacecolor= '#%02x%02x%02x' % tuple(color.as_rgb()), markersize=10))

        # relevances = np.around(relevances_sorted, decimals=3)

        images=[image, scene.astype("uint8")]
        nrows, ncols = (1, 3)

        if len(images) > nrows * ncols:
            raise ValueError(
                "The number of images exceeds the grid size. Please increase the grid size or reduce the number of images."
            )
        
        # relevances = np.squeeze(np.around(relevances_sorted, decimals=3), 1) 

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
        plt.tick_params(bottom=False, left=False)
        titles=['image', 'REVEAL']

        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                if images[idx].ndim == 2:
                    ax.imshow(images[idx], cmap="gray")
                    
                else:
                    ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
                    

                if titles is not None and idx < len(titles):
                    ax.set_title(titles[idx])
                    

            ax.axis("off")
        # l = ax.legend(custom_lines,[ "" for _ in range(len(custom_lines))] ,title=
        #               "The number of clusters identifyed is "+  str(len(masks)) +"\n",
        # loc='center left', bbox_to_anchor=(-0.17, 0.5), fancybox=True, shadow=True,  ncol=2,fontsize=11,
        # title_fontsize=11, alignment="center")

        l = ax.legend(custom_lines, relevances_sorted, title="CTC of " + label + "\n out of " + r"$\bf{" "%.3f" % full_relevance + "}$" "\n",
            loc='center left', bbox_to_anchor=(0.3, 0.5), fancybox=True, shadow=True, fontsize=11,
            title_fontsize=11, alignment="center")
        plt.savefig(image_path + "_inception"+ ".png")
        plt.setp(l.get_title(), multialignment='center')
        # plt.show()

        #        "Relevance of label \n " + label + "\n out of " + r"$\bf{" "%.3f" % full_relevance + "}$" "\n",
        # loc='center left', bbox_to_anchor=(0, 0.5), fancybox=True, shadow=True, fontsize=11,
        # title_fontsize=11, alignment="center")

    def mask_to_input_relevance_of_pixels(self, relevance, masks_from_heatmap3D, label, image_name): 
        print(relevance)
        # full_relevance = relevance[-1]
        # regions_relevance = relevance[-2]
        # relevances_clusters = relevance[:-2]
        # masks = masks_from_heatmap3D[:-2]
        relevances_clusters = relevance
        masks = masks_from_heatmap3D
        
        relevances_sorted, masks_with_ones_sorted = zip(*sorted(zip(relevances_clusters, masks), key = lambda x:x[0], reverse=True))

        masked_heat_show = np.zeros_like(masks_with_ones_sorted[0])
        custom_lines = []
        for i in range(len(masks_with_ones_sorted)):
            if relevances_sorted[i] > 0:
                mask = masks_with_ones_sorted[i]
                relevance = self.primes[i]
                masked_heat_show = masked_heat_show * np.logical_not(mask)
                masked_heat_show += relevance * mask
                # custom_lines.append(Line2D([0], [0], color= '#%02x%02x%02x' % tuple(color_map[relevance]), lw=1))
                custom_lines.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                    markerfacecolor= '#%02x%02x%02x' % tuple(self.color_map[relevance]), markersize=10))
            else:
                index = i - len(masks_with_ones_sorted)
                mask = masks_with_ones_sorted[index]
                relevance = self.primes[index]
                masked_heat_show = masked_heat_show * np.logical_not(mask)
                masked_heat_show += relevance * mask
                # custom_lines.append(Line2D([0], [0], color= '#%02x%02x%02x' % tuple(color_map[relevance]), lw=1))
                custom_lines.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                    markerfacecolor= '#%02x%02x%02x' % tuple(self.color_map[relevance]), markersize=10))

            # plt.imshow(masks[i], color=color_map[relevance * mask],
            #           ax=ax,
            #           label=mask_relevances[i])

        arr = masked_heat_show[:,:,:, 0][0]
        relevances_sorted = np.expand_dims(relevances_sorted, -1)

        data_3d = np.ndarray(shape=(arr.shape[0], arr.shape[1], 3), dtype=int)

        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                data_3d[i][j] = self.color_map[arr[i][j]]

        relevances = np.squeeze(np.around(relevances_sorted, decimals=3), 1) 

        fig, ax = plt.subplots()
        plt.tick_params(bottom=False, left=False)
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticklabels([])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # l = ax.legend(custom_lines, relevances, title="Relevance of label \n " + label + "\n out of " + r"$\bf{" "%.3f" % full_relevance + "}$" "\n",
        #             loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, fontsize=11,
        #             title_fontsize=11, alignment="center")

        # plt.setp(l.get_title(), multialignment='center')
        plt.imshow(data_3d)
        plt.savefig("vgg19/" + image_name + "_heatmap.png")
        # plt.show()
        plt.clf()

        # 
        # plt.show()              
