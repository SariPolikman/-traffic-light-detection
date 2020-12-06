import pickle
from PIL import Image
from model.tools.pls import PLS
import numpy as np

from model.traffic_lights.tfl_man import TflMan
from model import source


class Controller:
    def __init__(self, data_format: PLS, model=source.model):
        pkl_path, self.ids, self.frames = data_format.get_data_and_pls()

        with open(pkl_path, 'rb') as pkl_file:
            self.data = pickle.load(pkl_file, encoding='latin1')

        self.tfl_man = TflMan(model, self.data['flx'], self.data['principle_point'])

    def run(self):
        for ID, frame in zip(self.ids, self.frames):
            image = np.array(Image.open(frame))
            self.tfl_man.run(image, self.data[f'egomotion_{str(int(ID) - 1)}-{ID}'])
