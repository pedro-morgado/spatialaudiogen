from skimage import io
import os

PWD = os.path.dirname(__file__)


def poodle():
    return io.imread(os.path.join(PWD, 'poodle.png'))


def laska():
    return io.imread(os.path.join(PWD, 'laska.png'))


def cat():
    return io.imread(os.path.join(PWD, 'cat.jpg'))


def dog():
    return io.imread(os.path.join(PWD, 'dog.png'))


def puzzle():
    return io.imread(os.path.join(PWD, 'puzzle.jpeg'))


def tiger():
    return io.imread(os.path.join(PWD, 'tiger.jpeg'))
