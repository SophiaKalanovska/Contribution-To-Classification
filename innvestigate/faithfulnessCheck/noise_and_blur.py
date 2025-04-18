import numpy as np
import cv2
from PIL import Image
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=0.01):
    """
    Adds Gaussian noise to an image.

    :param image: A NumPy array representing the image.
    :param mean: Mean of the Gaussian noise.
    :param sigma: Standard deviation of the Gaussian noise.
    :return: Image with Gaussian noise added.
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are within proper range
    return noisy_image.astype(np.uint8)




def gaussian_blur(image, kernel_size=(5, 5)):
    """
    Applies Gaussian blur to an image.

    :param image: A NumPy array representing the image.
    :param kernel_size: Size of the Gaussian kernel.
    :return: Blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)



def add_uniform_noise(image, noise_level=0.3):
    """
    Add random noise to an image.
    :param image_path: Path to the input image.
    :param noise_level: The intensity of the noise to be added, between 0 and 1.
    :return: A PIL Image object with noise added.
    """

    # Convert the original image to a NumPy array
    original_image_np = np.array(image)
    # Generate random noise
    noise = np.random.uniform(-255 * noise_level, 255 * noise_level, original_image_np.shape).astype(np.uint8)
    # Add the noise to the original image and clip to ensure values are between 0 and 255
    noisy_image_np = np.clip(original_image_np + noise, 0, 255)
    # Convert back to PIL Image for output
    # noisy_image = Image.fromarray(noisy_image_np.astype(np.uint8))
    return noisy_image_np.astype(np.uint8)