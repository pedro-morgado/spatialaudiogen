from common import spherical_harmonics_matrix
from pyutils.ambisonics.position import Position
import numpy as np

DECODING_METHODS = ['projection', 'pseudoinv']
DEFAULT_DECODING = 'projection'


class AmbiDecoder(object):
    def __init__(self, speakers_pos, ambi_format, method=DEFAULT_DECODING):
        assert method in DECODING_METHODS
        if isinstance(speakers_pos, Position):
            speakers_pos = [speakers_pos]
        assert isinstance(speakers_pos, list) and all([isinstance(p, Position) for p in speakers_pos])
        self.speakers_pos = speakers_pos
        self.sph_mat = spherical_harmonics_matrix(speakers_pos,
                                                  ambi_format.order,
                                                  ambi_format.ordering,
                                                  ambi_format.normalization)
        self.method = method
        if self.method == 'pseudoinv':
            self.pinv = np.linalg.pinv(self.sph_mat)

    def decode(self, ambi):
        if self.method == 'projection':
            return np.dot(ambi, self.sph_mat.T)
        if self.method == 'pseudoinv':
            return np.dot(ambi, self.pinv)
