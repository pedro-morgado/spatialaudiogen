from math import cos, sin, atan2, sqrt, pi
import numpy as np


class Position(object):
    def __init__(self, x1, x2, x3, c_type):
        assert c_type.lower() in ['cartesian', 'polar']

        self.x, self.y, self.z = 0., 0., 0.
        self.phi, self.nu, self.r = 0., 0., 0.
        if c_type == 'cartesian':
            self.set_cartesian(x1, x2, x3)
        else:
            self.set_polar(x1, x2, x3)

    def clone(self):
        return Position(self.x, self.y, self.z, 'cartesian')

    def set_cartesian(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.calc_polar()
        self.calc_cartesian()

    def set_polar(self, phi, nu, r):
        self.phi, self.nu, self.r = float(phi), float(nu), float(r)
        self.calc_cartesian()
        self.calc_polar()

    def calc_cartesian(self):
        self.x = self.r * cos(self.phi) * cos(self.nu)
        self.y = self.r * sin(self.phi) * cos(self.nu)
        self.z = self.r * sin(self.nu)

    def calc_polar(self):
        self.phi = atan2(self.y, self.x)
        self.nu = atan2(self.z, sqrt(self.x**2+self.y**2))
        self.r = sqrt(self.x**2+self.y**2+self.z**2)

    def rotate(self, rot_matrix):
        pos = np.dot(rot_matrix, np.array([self.x, self.y, self.z]).reshape(3, 1))
        self.x, self.y, self.z = pos[0], pos[1], pos[2]
        self.calc_polar()
        self.calc_cartesian()

    def set_radius(self, radius):
        self.r = radius
        self.calc_cartesian()

    def coords(self, c_type):
        if c_type == 'cartesian':
            return np.array([self.x, self.y, self.z])
        elif c_type == 'polar':
            return np.array([self.phi, self.nu, self.r])
        else:
            raise ValueError, 'Unknown coordinate type. Use cartesian or polar.'

    def print_position(self, c_type=None):
        if c_type is None or c_type == 'cartesian':
            print('Cartesian (x,y,z): (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z))
        if c_type is None or c_type == 'polar':
            print('Polar (phi,nu,r):  (%.2f, %.2f, %.2f)' % (self.phi, self.nu, self.r))


class PositionalSource(object):
    def __init__(self, signal, position, sample_rate=44800):
        assert not isinstance(position, list)
        assert signal.ndim == 1
        self.signal = signal
        self.position = position
        self.sample_rate = sample_rate


class MovingSource(PositionalSource):
    def __init__(self, signal, positions, rate=44800):
        super(MovingSource, self).__init__(signal, Position(0, 0, 0, 'polar'), rate)
        # PositionalSource.__init__(self, signal, Position(0, 0, 0, 'polar'), rate)

        duration = signal.shape[0] / float(rate)
        self.pts_p = positions
        self.npts = len(self.pts_p)
        self.pts_t = np.linspace(0, duration, self.npts)
        self.nframes = int(duration * rate)
        self.dt = 1/float(rate)

        self.pts_idx = np.floor(np.linspace(0, (self.npts-1), self.nframes)).astype(int)
        self.cur_idx = -1

    def tic(self):
        if self.cur_idx >= (self.nframes-1):
            return False

        self.cur_idx += 1
        cur_t = self.cur_idx * self.dt
        idx = self.pts_idx[self.cur_idx]
        if idx == (self.npts-1):
            self.position = self.pts_p[-1]
        else:
            alpha = (cur_t - self.pts_t[idx]) / (self.pts_t[idx+1] - self.pts_t[idx])
            cur_pos = alpha * self.pts_p[idx + 1].coords('polar') + \
                      (1 - alpha) * self.pts_p[idx].coords('polar')
            self.position.set_polar(cur_pos[0], cur_pos[1], cur_pos[2])
        return True


def test_moving_source():
    from pyutils.iolib.audio import load_wav
    mono, rate = load_wav('wav_test/piano.wav')
    mono = mono[:, 0]

    position_fn = 'wav_test/piano_mov_position.txt'
    positions = [np.array([float(num) for num in l.strip().split()]) for l in open(position_fn, 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]

    source = MovingSource(mono, positions, rate)
    while source.tic():
        source.position.print_position('polar')


def test_position():
    p = Position(-pi/4, pi/4, 1, 'polar')
    p.print_position('polar')

if __name__ == '__main__':
    # test_position()
    test_moving_source()
