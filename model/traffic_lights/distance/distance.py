from model.traffic_lights.distance.data_pls import DataPls


class TflDistances:
    def __init__(self, focal, pp):
        self.data = DataPls(focal, pp)
