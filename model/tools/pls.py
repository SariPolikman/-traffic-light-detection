class PLS:
    def __init__(self):
        pass

    def get_data_and_pls(self):
        pass


class File(PLS):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def get_data_and_pls(self):  # TODO test cases ,can open pkl file here.... its better?
        with open(self.file, 'r', encoding='UTF8') as f:
            lines = f.read().splitlines()
        return lines[0], lines[1::2], lines[2::2]
