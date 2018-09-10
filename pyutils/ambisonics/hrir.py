from math import sin, cos, pi, sqrt

import numpy as np
import os
from common import Position
from pyutils.iolib.audio import load_wav
from sklearn.neighbors import KDTree


class CIPIC_HRIR(object):
    def __init__(self, dirname):
        elevation = np.array([-45, -39, -34, -28, -23, -17, -11, -6, 0, 6, 11, 17, 23, 28, 34, 39, 45, 51, 56, 62, 68, 73, 79, 84, 90, 96, 101, 107, 113, 118, 124, 129, 135, 141, 146, 152, 158, 163, 169, 174, 180, 186, 191, 197, 203, 208, 214, 219, 225, 231])
        azimuth = np.array([-80, -65, -55, -45, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 45, 55, 65, 80])

        self.right_hrir = np.zeros((200, len(azimuth), len(elevation)))
        self.left_hrir = np.zeros((200, len(azimuth), len(elevation)))
        for i, phi in enumerate(azimuth):
            right_fn = ('neg' if phi < 0 else '') + str(abs(phi)) + 'azright.wav'
            left_fn = ('neg' if phi < 0 else '') + str(abs(phi)) + 'azleft.wav'
            self.right_hrir[:, i, :] = np.flip(load_wav(os.path.join(dirname, right_fn))[0], axis=0)
            self.left_hrir[:, i, :] = np.flip(load_wav(os.path.join(dirname, left_fn))[0], axis=0)

        radius = 3.
        self.hrir_db = []
        for i, az in enumerate(azimuth):
            for j, elev in enumerate(elevation):
                xp = radius * cos(elev*pi/180.) * sin(az*pi/180.)
                yp = radius * cos(elev*pi/180.) * cos(az*pi/180.)
                zp = radius * sin(elev*pi/180.)
                x, y, z = yp, -xp, zp
                # x, y, z = xp, yp, zp
                p = Position(x, y, z, 'cartesian')
                self.hrir_db.append((p, self.left_hrir[:, i, j], self.right_hrir[:, i, j]))

        self.kdt = KDTree(np.array([hrir[0].coords('cartesian') / np.linalg.norm(hrir[0].coords('cartesian'))
                                    for hrir in self.hrir_db]), leaf_size=2, metric='euclidean')

    def get_closest(self, pos):
        assert isinstance(pos, Position)
        i = self.kdt.query(np.array([pos.x, pos.y, pos.z]).reshape((1, -1))/sqrt(pos.x**2+pos.y**2+pos.z**2))[1][0, 0]
        return self.hrir_db[i]


def test_cipic_hrir():
    hrir = CIPIC_HRIR('hrtfs/cipic_subj3')
    hrir.get_closest(Position(7*pi/4, -pi/4, 3, 'polar'))

if __name__ == '__main__':
    test_cipic_hrir()
