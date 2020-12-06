import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage

RED = 0
GREEN = 1

N = 150


def init_kernel_red():
    return np.array([[-1 / 9, -1 / 9, -1 / 9],
                     [-1 / 9, 8 / 9, -1 / 9],
                     [-1 / 9, -1 / 9, -1 / 9]])


def init_kernel_green():
    return np.array([[-1 / 9, -1 / 9, -1 / 9],
                     [-1 / 9, 8 / 9, -1 / 9],
                     [-1 / 9, -1 / 9, -1 / 9]])


def print_image(image):
    img = Image.fromarray(image, 'RGB')
    img.show()


def filter_by_arg(max_filtered, grad, arg):
    _filter = np.argwhere(grad > arg)
    max_filtered = np.argwhere(max_filtered == grad)
    _filter = set((tuple(i) for i in _filter))
    max_filtered = set((tuple(i) for i in max_filtered))

    res = _filter.intersection(max_filtered)

    x = list(map(lambda x: x[0], res))
    y = list(map(lambda x: x[1], res))

    return x, y


def find_n_max_pixels(c_image):
    # norm_image = c_image/255
    # print(norm_image)
    pixel = np.sort(c_image.ravel())
    # print("somme....",pixel[0], pixel[-1], pixel[-N],pixel[N])
    len_ = len(pixel)
    range_ = abs(pixel[-1]) - abs(pixel[0]) if abs(pixel[-1]) > abs(pixel[0]) else abs(pixel[0]) - abs(pixel[-1])
    distribution = len_ / range_
    # print("len", len_)
    # print("range", range_)
    # print("distribution", distribution)
    # print("pixel",pixel)
    # print("try min",np.argwhere(c_image < pixel[0]))
    # print("try max",np.argwhere(c_image > pixel[-1]))

    # print("pixel",pixel)
    # return pixel[-N]
    return 29


class TFL:

    def __init__(self, image=None):
        self.kernel = dict()
        self.kernel[GREEN] = init_kernel_green()
        self.kernel[RED] = init_kernel_red()
        self.image = image

    def convolve_by_color(self, image, color):
        gray_image = image[:, :, color]
        grad = signal.convolve2d(gray_image.T, self.kernel[color], boundary='symm', mode='same')  # TODO check pading

        return grad

    def height(self):
        return self.image.shape[0]

    def weight(self):
        return self.image.shape[1]


def print_convolve(image, grad):
    fig, (ax_orig, ax_mag) = plt.subplots(2, 1, figsize=(6, 15))

    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_axis_off()

    ax_mag.imshow(np.absolute(grad.T), cmap='gray')
    ax_mag.set_axis_off()

    fig.show()


def find_tfl_lights_by_color(tfl, color):
    grad = tfl.convolve_by_color(tfl.image, color)

    g = ndimage.maximum_filter(grad, size=20)

    max_pixels = find_n_max_pixels(grad)

    x, y = filter_by_arg(g, grad, max_pixels)

    return x, y


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    tfl = TFL(c_image)

    green_x, green_y = find_tfl_lights_by_color(tfl, GREEN)
    red_x, red_y = find_tfl_lights_by_color(tfl, RED)
    return list(zip(red_y, red_x)) + list(zip(green_y, green_x)), ("R" + (len(red_x) - 1) * " R" + len(green_x) * " G").split(" ")
