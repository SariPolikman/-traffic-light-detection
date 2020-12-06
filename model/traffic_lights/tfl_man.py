from view import plot
import matplotlib.pyplot as plt
from model.traffic_lights.filter_light.filter_light import FilterLight
from model.traffic_lights.distance.distance import TflDistances
from model.traffic_lights.distance.frame_data import FrameData
from typing import List

from model.tools.crop_image import crop_all_images
from model.traffic_lights.detect_light import detect_light
from model.traffic_lights.distance.SFM_standAlone import get_distance


class TflMan:
    def __init__(self, model, focal, pp):
        self.filter_light = FilterLight(model)
        self.tfl_distances = TflDistances(focal, pp)
        self.prev = FrameData(None, None)

    def run(self, image, EM):
        fig, (light_src, tfl, distances) = plt.subplots(1, 3, figsize=(20, 8))
        data = FrameData(image, EM)

        data.coordination, data.auxiliary = detect_light.find_tfl_lights(data.image)

        assert len(data.coordination) == len(data.auxiliary)

        if not data.coordination:  # no lights
            return

        plot.mark_tfl(image, data.coordination, data.auxiliary, light_src, "detect light")

        coordination, auxiliary = self.filter_tfl(tfl, data.image, data.coordination, data.auxiliary)

        assert len(coordination) <= len(data.coordination)

        if len(coordination) <= len(data.coordination):
            data.coordination = coordination
            data.auxiliary = auxiliary
        else:
            # if image disturb use prev frame
            data = self.prev

        plot.mark_tfl(data.image, data.coordination, data.auxiliary, tfl, "filter tfl")

        if not self.prev.coordination:
            self.prev = data
            plt.show()
            return

        if not len(data.coordination):  # no tfl's
            self.prev = data
            plot.show_img(data.image, "distances (no tfl's)", distances)
            plt.show()
            return

        data.distance = get_distance(distances, self.prev.image, data.image, self.prev.coordination,
                                     data.coordination, data.EM, self.tfl_distances.data.focal, self.tfl_distances.data.pp)
        self.prev = data

        plt.show()

    def filter_tfl(self, fig, image, candidates: List[int], auxiliary: List[int]) -> (List[int], List[int]):
        assert len(candidates) == len(auxiliary)

        images = crop_all_images(image, candidates)
        predictions = self.filter_light.predict_model(images)

        relevant = [c for i, c in enumerate(candidates) if predictions[:, 1][i] > 0.97]  # TODO by filter
        aux = [a for i, a in enumerate(auxiliary) if predictions[:, 1][i] > 0.97]

        return relevant, aux
