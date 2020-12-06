from model import source
from model.tools.pls import File
from controller.controller import Controller


def main():
    file = File(source.file)
    controller = Controller(file)
    controller.run()


if __name__ == '__main__':
    main()
