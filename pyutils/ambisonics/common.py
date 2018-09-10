from pyutils.ambisonics.position import Position
from math import sqrt, pi, factorial
import numpy as np

CHANNEL_ORDERING = ['FURSE_MALHAM', 'SID', 'ACN']
NORMALIZATION = ['MAX_N', 'SN3D', 'N3D']
DEFAULT_ORDERING = 'ACN'
DEFAULT_NORMALIZATION = 'SN3D'
DEFAULT_RATE = 44100
DEFAULT_RADIUS = 1.
DEFAULT_ORDER = 1


class AmbiFormat(object):
    def __init__(self,
                 ambi_order=DEFAULT_ORDER,
                 sample_rate=DEFAULT_RATE,
                 radius=DEFAULT_RADIUS,
                 ordering=DEFAULT_ORDERING,
                 normalization=DEFAULT_NORMALIZATION):
        self.order = ambi_order
        self.num_channels = int((ambi_order+1)**2)
        self.radius = radius
        self.sample_rate = sample_rate
        self.ordering = ordering
        self.normalization = normalization


class AmbisonicArray(object):
    def __init__(self, data, ambi_format=AmbiFormat()):
        self.data = data
        self.format = ambi_format

    def convert(self, sample_rate=None, ordering=None, normalization=None):
        assert sample_rate is not None or ordering is not None or normalization is not None
        n = self.format.num_channels

        if sample_rate is not None and sample_rate != self.format.sample_rate:
            from scipy.signal import resample
            duration = float(self.data.shape[0]) / self.format.sample_rate
            data = resample(self.data, int(duration * sample_rate))
            self.format.sample_rate = sample_rate
        else:
            data = np.copy(self.data)

        if ordering is not None and ordering != self.format.ordering:
            assert ordering in CHANNEL_ORDERING
            mapping = map(lambda x: convert_ordering(x, ordering, self.format.ordering), range(n))
            data = data[:, mapping]
            self.format.ordering = ordering

        if normalization is not None and normalization != self.format.normalization:
            assert normalization in NORMALIZATION
            c_out = np.array(map(lambda x: normalization_factor(x, self.format.ordering, normalization), range(n)))
            c_in = np.array(map(lambda x: normalization_factor(x, self.format.ordering, self.format.normalization), range(n)))
            data *= (c_out / c_in).reshape((1, -1))
            self.format.normalization = normalization

        self.data = data


def degree_order_to_index(order, degree, ordering=DEFAULT_ORDERING):
    assert -order <= degree <= order
    assert ordering in CHANNEL_ORDERING

    def acn_idx(n, m):
        return n*(n+1)+m

    def sid_idx(n, m):
        idx_order = [1+i*2 for i in range(n)] + [n*2] + list(reversed([i*2 for i in range(n)]))
        return idx_order[m+n] + n**2

    def fm_idx(n, m):
        if n == 1:
            idx_order = [1, 2, 0]
        else:
            idx_order = list(reversed([2*(i+1) for i in range(n)])) + [0] + [1+i*2 for i in range(n)]
        return idx_order[m+n] + n**2

    if ordering == 'ACN':
        return acn_idx(order, degree)
    elif ordering == 'FURSE_MALHAM':
        return fm_idx(order, degree)
    else:
        return sid_idx(order, degree)


def index_to_degree_order(index, ordering=DEFAULT_ORDERING):
    assert ordering in CHANNEL_ORDERING
    order = int(sqrt(index))
    index -= order**2
    if ordering == 'ACN':
        degree = index - order
        return order, degree
    elif ordering == 'FURSE_MALHAM':
        if order == 1:
            mapping = [1, -1, 0]
            degree = mapping[index]
        else:
            degree = (int(index)+1)/2
            if index % 2 == 0:
                degree = -degree
        return order, degree
    else:
        degree = (int(index)+1)/2
        if index % 2 == 0:
            degree = -degree
        return order, degree


def convert_ordering(index, orig_ordering, dest_ordering):
    assert orig_ordering in CHANNEL_ORDERING
    assert dest_ordering in CHANNEL_ORDERING
    if dest_ordering == orig_ordering:
        return index

    n, m = index_to_degree_order(index, orig_ordering)
    return degree_order_to_index(n, m, dest_ordering)


def normalization_factor(index, ordering=DEFAULT_ORDERING, normalization=DEFAULT_NORMALIZATION):
    assert ordering in CHANNEL_ORDERING
    assert normalization in NORMALIZATION

    def max_norm(n, m):
        assert n <= 3
        if n == 0:
            return 1/sqrt(2.)
        elif n == 1:
            return 1.
        elif n == 2:
            return 1. if m == 0 else 2. / sqrt(3.)
        else:
            return 1. if m == 0 else (sqrt(45. / 32) if m in [1, -1] else 3. / sqrt(5.))

    def sn3d_norm(n, m):
        return sqrt((2. - float(m == 0)) * float(factorial(n-abs(m))) / float(factorial(n+abs(m))))

    def n3d_norm(n, m):
        return sn3d_norm(n, m) * sqrt((2*n+1) / (4.*pi))

    order, degree = index_to_degree_order(index, ordering)
    if normalization == 'MAX_N':
        return max_norm(order, degree)
    elif normalization == 'N3D':
        return n3d_norm(order, degree)
    elif normalization == 'SN3D':
        return sn3d_norm(order, degree)


def spherical_harmonic_mn(order, degree, phi, nu, normalization=DEFAULT_NORMALIZATION):
    from scipy.special import lpmv
    norm = normalization_factor(degree_order_to_index(order, degree), normalization=normalization)
    sph = (-1)**degree * norm * \
        lpmv(abs(degree), order, np.sin(nu)) * \
        (np.cos(abs(degree) * phi) if degree >= 0 else np.sin(abs(degree) * phi))
    return sph


def spherical_harmonics(position, max_order, ordering=DEFAULT_ORDERING, normalization=DEFAULT_NORMALIZATION):
    assert isinstance(position, Position)

    num_channels = int((max_order+1)**2)
    output = np.zeros((num_channels,))
    for i in range(num_channels):
        order, degree = index_to_degree_order(i, ordering)
        output[i] = spherical_harmonic_mn(order, degree, position.phi, position.nu, normalization)
    return output


def spherical_harmonics_matrix(positions, max_order, ordering=DEFAULT_ORDERING, normalization=DEFAULT_NORMALIZATION):
    assert isinstance(positions, list) and all([isinstance(p, Position) for p in positions])

    num_channels = int((max_order + 1) ** 2)
    Y = np.zeros((len(positions), num_channels))
    for i, p in enumerate(positions):
        Y[i] = spherical_harmonics(p, max_order, ordering, normalization)
    return Y
