import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from import_data import Import_from_files

length = 81
width = 81
tfl_color = 19
car_color = 26
tree_color = 21
signpost_color = 17

train_city = r"\train_try"
val_city = r"\frankfurt"


def load_data(bin_file):
    data = np.fromfile(bin_file, dtype='uint8')
    sum_img = len(data) // (length * width * 3)
    data = data.reshape(sum_img, length, width, 3)
    return data


def save_to_bin_file(data, path):
    np.array(data, dtype='uint8').tofile(path)


def show_image_and_label(dir):
    labels_in_words = {0: "No TFL", 1: "TFL"}

    images = np.fromfile(f'data_dir/{dir}/data.bin', dtype='uint8')
    num_of_imgs = len(images) // (81 * 81 * 3)
    images = images.reshape(num_of_imgs, 81, 81, 3)

    labels = np.fromfile(f'data_dir/{dir}/label.bin', dtype='uint8')
    print(labels)
    print(len(labels))
    print(num_of_imgs)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        pos = np.random.randint(num_of_imgs)
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[pos], cmap=plt.cm.binary)
        plt.xlabel(labels_in_words[labels[pos]])
    plt.show()


class Dataset:
    def __init__(self, import_format):
        self.data = {}
        self.crop_array = {"train": [], "val": []}
        self.Import = import_format()

    def import_data(self):
        data = {"train": self.get_img_label("train"),
                "val": self.get_img_label("val")}
        return data

    def get_img_label(self, type_path, d=""):

        dict_ = {"imgs": self.Import.import_img(type_path, d),
                 "labels": self.Import.import_label(type_path, d)}
        return dict_
        # dict_ = {}
        # dict_["imgs"], dict_["labels"]= read_dirs(type_path)
        # return dict_

    def dispatch_dict(self, operator, x, y):
        return {
            '==': lambda: x == y,
            '!=': lambda: x != y,
        }.get(operator, lambda: None)()

    def get_rand_idx(self, lst):
        return random.randint(0, len(lst) - 1)

    def get_rand_coordinates_by_filter(self, img, opt, arg):
        cond = self.dispatch_dict(opt, img, arg)
        coordinates = np.argwhere(cond)
        if coordinates.size <= 0:
            return []
        rand_pixel = self.get_rand_idx(coordinates)
        return coordinates[rand_pixel]

    def get_relevant_coordinate_by_filter(self, label, opt, arg):
        image_pad = np.pad(label, ((40, 41), (40, 41)), 'constant')
        pixel = self.get_rand_coordinates_by_filter(label, opt, arg)
        while len(pixel) > 0:

            sub_img = image_pad[pixel[0]: pixel[0] + length, pixel[1]: pixel[1] + width]
            if len(self.get_rand_coordinates_by_filter(sub_img, "==", tfl_color)) >= 0:
                break
            pixel = self.get_rand_coordinates_by_filter(label, "!=", tfl_color)
        return pixel

    def open_image(self, img):
        return np.asarray(Image.open(img))

    def get_pair_tfl_and_not(self, img):
        """For each example where a traffic light exists, use the labels to pick one
        pixel that falls on a traffic light and one random pixel that does not fall on a
        traffic light"""
        label = self.open_image(img)
        pixel_19 = [self.get_rand_coordinates_by_filter(label, "==", tfl_color) for _ in range(5)]
        pixel_ne_19 = [self.get_relevant_coordinate_by_filter(label, "==", car_color) for _ in range(2)]
        pixel_ne_19.append(self.get_relevant_coordinate_by_filter(label, "!=", tfl_color))
        pixel_ne_19.append(self.get_relevant_coordinate_by_filter(label, "==", tree_color))
        pixel_ne_19.append(self.get_relevant_coordinate_by_filter(label, "==", signpost_color))

        # if there are no tfl's\cars... in the image
        pixel_19 = list(filter([].__ne__, pixel_19))
        pixel_ne_19 = list(filter([].__ne__, pixel_ne_19))
        return pixel_19, pixel_ne_19

    def crop(self, img, pixel, type_):
        """Crop the images around your chosen pixels"""
        image = self.open_image(img)

        image_pad = np.pad(image, ((40, 41), (40, 41), (0, 0)), mode='constant')
        sub_image = image_pad[pixel[0]: pixel[0] + 81, pixel[1]: pixel[1] + 81]
        sub_tfl_list = sub_image.tolist()
        self.crop_array[type_].append(sub_tfl_list)
        return sub_image

    def prepare_dataset(self, type_):
        l = len(self.data[type_]["labels"])
        s = 1
        for label, img in zip(self.data[type_]["labels"], self.data[type_]["imgs"]):
            print(f'{s}/{l}')
            s += 1
            yes_idx, no_idx = self.get_pair_tfl_and_not(label)
            for y, n in zip(yes_idx, no_idx):
                self.crop(img, y, type_)
                self.crop(img, n, type_)

    def save(self, dir):
        """ Save data """
        tmp = np.array(self.crop_array[dir])
        save_to_bin_file(tmp, f"data_dir/{dir}/data.bin")
        _list = [1, 0] * (len(self.crop_array[dir]) // 2)
        labels = np.array(_list)
        save_to_bin_file(labels, f"data_dir/{dir}/label.bin")

    def show_image(self, img, title=""):
        plt.imshow(img)
        plt.title(title)
        plt.show()

    def display(self, path):
        """ Make a function to display data from your dataset with the
        corresponding label"""
        labels = np.fromfile(f'{path}/label.bin', dtype='uint8')
        data = load_data(f'{path}/data.bin')
        print(len(labels))
        print(len(data))
        for img, label in zip(data, labels):
            title = 'yes TFL' if label % 2 else 'no TFL'
            self.show_image(img, title)

    def display_random_imgs(self, path):
        labels = np.fromfile(f'{path}/label.bin', dtype='uint8')
        data = load_data(f'{path}/data.bin')
        num_of_imgs = len(data) // (81 * 81 * 3)
        images = data.reshape(num_of_imgs, 81, 81, 3)

        plt.figure(figsize=(10, 10))
        for i in range(25):
            pos = np.random.randint(num_of_imgs)
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[pos], cmap=plt.cm.binary)
            plt.xlabel(title='yes TFL' if labels[pos] % 2 else 'no TFL')
        plt.show()


def building_dataset():
    dataset = Dataset(Import_from_files)
    dataset.data = dataset.import_data()
    dataset.prepare_dataset("train")
    dataset.prepare_dataset("val")
    dataset.save("train")
    dataset.save("val")
    return dataset


def main():
    dataset = building_dataset()
    dataset.display('data_dir/val')
    show_image_and_label("train")


if __name__ == '__main__':
    main()
