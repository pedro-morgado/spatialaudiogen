import numpy as np
from common import AmbisonicArray, spherical_harmonics_matrix
from position import PositionalSource


class AmbiEncoder(object):
    def __init__(self, ambi_format):
        self.format = ambi_format

    def encode(self, sources):
        if isinstance(sources, PositionalSource):
            sources = [sources]
        assert isinstance(sources, list) and all([isinstance(src, PositionalSource) for src in sources])
        assert all([src.signal.ndim == 1 for src in sources])

        fmt = self.format
        Y = spherical_harmonics_matrix([src.position for src in sources],
                                       fmt.order,
                                       fmt.ordering,
                                       fmt.normalization)
        src_signals = np.stack([src.signal for src in sources], axis=1)
        return AmbisonicArray(np.dot(src_signals, Y), self.format)

    def encode_frame(self, sources, ambi_array, frame_no):
        if isinstance(sources, PositionalSource):
            sources = [sources]
        assert isinstance(sources, list) and all([isinstance(src, PositionalSource) for src in sources])

        Y = spherical_harmonics_matrix([src.position for src in sources],
                                       self.format.order,
                                       self.format.ordering,
                                       self.format.normalization)
        src_signal = np.array([src.signal[frame_no] for src in sources]).reshape((-1, 1))
        ambi_array.data[frame_no] = (src_signal * Y).sum(axis=0)

    def encode_v2(self, sources):
        from scipy.ndimage.interpolation import shift
        if isinstance(sources, PositionalSource):
            sources = [sources]
        assert isinstance(sources, list) and all([isinstance(src, PositionalSource) for src in sources])
        radius = self.format.radius
        if radius > 0:
            assert all([src.position.r > radius for src in sources])

        # Reposition sources on the sphere. Account for delays and attenuation
        for src in sources:
            if src.position.r > 0:
                dist = src.position.r - radius
                delay = int(dist / 343. * src.sample_rate)
                attenuation = 1. / (1. + dist)
                src.signal = shift(src.signal, delay, cval=0.) * attenuation
                src.position.set_radius(radius)

        # Encode ambisonics
        return self.encode(sources)


