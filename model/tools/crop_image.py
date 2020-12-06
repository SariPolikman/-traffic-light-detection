import numpy as np


def crop(image, pixel):
    """ Crop the images around your chosen pixels """

    image_pad = np.pad(image, ((40, 41), (40, 41), (0, 0)), mode='constant')
    sub_image = image_pad[pixel[0]: pixel[0] + 81, pixel[1]: pixel[1] + 81]
    return sub_image


def crop_all_images(img, candidates):
    cropped_images = []
    for c in candidates:
        cropped = crop(img, c)
        cropped_images.append(cropped)
        assert cropped.shape == (81, 81, 3)

    return np.array(cropped_images)
