import random

import numpy
from pandas import np
from scipy.ndimage import rotate

from build_dataset import load_data, save_to_bin_file, show_image_and_label


class Augmentation:
    def __init__(self, bin_data_file, bin_labels_file):
        self.data = load_data(bin_data_file)
        self.labels = np.fromfile(bin_labels_file, dtype='uint8')

    def rotate_data(self, degrees):

        counter = 0
        rotates = []
        for image in self.data:
            rand_degrees = random.uniform(10, degrees)
            rot = rotate(image, rand_degrees, reshape=False)
            print(rand_degrees)
            print("rotate", counter)
            counter += 1
            rotates.append(rot)

        return rotates

    def mirror_data(self):
        mirrors = []
        counter = 0
        for image in self.data:
            print("mirror", counter)
            counter += 1
            mirror = image[:, ::-1, :]
            mirrors.append(mirror)

        return mirrors

    def save(self, data, label, dir):
        imgs = np.array(data)
        save_to_bin_file(imgs, f"data_dir/{dir}/data.bin")
        save_to_bin_file(label, f"data_dir/{dir}/label.bin")


    def display(self, path):
        """ Make a function to display data from your dataset with the
        corresponding label"""
        labels = np.fromfile(f'{path}/label.bin', dtype='uint8')
        data = load_data(f'{path}/data.bin')
        for img, label in zip(data, labels):
            title = 'yes TFL' if label % 2 else 'no TFL'
            self.show_image(img, title)

    def merge_files(self, file1_path, file2_path, dir):
        data1 = load_data(f'{file1_path}/data.bin')
        print("data1 typpe", data1.shape)
        data2 = load_data(f'{file2_path}/data.bin')
        label1 = np.fromfile(f'{file1_path}/label.bin', dtype='uint8')
        label2 = np.fromfile(f'{file2_path}/label.bin', dtype='uint8')
        data = numpy.concatenate([data1, data2])
        print("data type", data.shape)
        print("label1 type", label1.shape)
        labels = numpy.concatenate([label1, label2])
        print("labels type", labels.shape)
        self.save(data, labels, dir)

    def zoom_data(self, zoom):
        zm1 = clipped_zoom(img, 0.5)
        zm2 = clipped_zoom(img, 1.5)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        ax[1].imshow(zm1)
        ax[2].imshow(zm2)


def augmentations():
    augmentation = Augmentation('data_dir/train/data.bin', 'data_dir/train/label.bin')

    rotate = augmentation.rotate_data(60)
    mirror = augmentation.mirror_data()
    augmentation.save(rotate, augmentation.labels, "rotate1")
    show_image_and_label("rotate1")
    augmentation.save(mirror, augmentation.labels, "mirror")

    augmentation.merge_files('data_dir/mirror', "data_dir/train", "b_mirror")
    show_image_and_label("mirror")
