import os

basic_path = r"C:\Users\RENT\Desktop\CityScapes"
labels_path = "gtFine"
img_path = "leftImg8bit"
train_city = r"\train_try"
val_city = r"\frankfurt"


class Import_from_files:
    def __init__(self):
        pass

    def import_img(self, type_path, d=""):
        data = []
        for path, _, files in os.walk(f"{basic_path}/{img_path}/{type_path}{d}"):
            for file in files:
                # img = np.asarray(Image.open(f"{path}/{file}"))
                data.append(f'{path}/{file}')
        return data

    def import_label(self, type_path, d=""):
        data = []
        for path, _, files in os.walk(f"{basic_path}/{labels_path}/{type_path}{d}"):
            for file in files:
                if file.endswith("gtFine_labelIds.png"):
                    # img = np.asarray(Image.open(f"{path}/{file}"))
                    data.append(f'{path}/{file}')
        return data
