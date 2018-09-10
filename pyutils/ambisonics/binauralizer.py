import os
import numpy as np
from decoder import AmbiDecoder
from hrir import CIPIC_HRIR
from pyutils.ambisonics.position import Position, PositionalSource, MovingSource
from scipy.ndimage.interpolation import shift
from tdesigns import get_tDesign

C = 343.     # Speed of sound [m/s]


class VirtualStereoMic(object):
    def __init__(self, radius=0.1):
        self.radius = radius
        self.lmic_pos = Position(0, radius, 0, 'cartesian')
        self.rmic_pos = Position(0, -radius, 0, 'cartesian')

    def binauralize(self, sources):
        if isinstance(sources, PositionalSource):
            sources = [sources]

        l_signal, r_signal = 0, 0.
        for src in sources:
            l_dist = np.sqrt(((src.position.coords('cartesian') - self.lmic_pos.coords('cartesian'))**2).sum())
            r_dist = np.sqrt(((src.position.coords('cartesian') - self.rmic_pos.coords('cartesian'))**2).sum())

            # Time delay
            l_delay, r_delay = int(l_dist / C * src.sample_rate), int(r_dist / C * src.sample_rate)

            # Attenuation is frequency dependent, but lets simplify.
            l_attn, r_attn = 1 / (1. + l_dist), 1 / (1. + r_dist)

            l_signal += l_attn * shift(src.signal, l_delay, cval=0.) / len(sources)
            r_signal += r_attn * shift(src.signal, r_delay, cval=0.) / len(sources)

        return np.stack((l_signal, r_signal), axis=1)

    def binauralize_frame(self, sources, output, frame_no):
        if isinstance(sources, PositionalSource):
            sources = [sources]

        for src in sources:
            l_dist = np.sqrt(((src.position.coords('cartesian') - self.lmic_pos.coords('cartesian'))**2).sum())
            r_dist = np.sqrt(((src.position.coords('cartesian') - self.rmic_pos.coords('cartesian'))**2).sum())

            # Time delay
            l_delay, r_delay = int(l_dist / C * src.sample_rate), int(r_dist / C * src.sample_rate)

            # Attenuation is frequency dependent, but lets simplify.
            l_attn, r_attn = 1 / (1. + l_dist), 1 / (1. + r_dist)

            if frame_no-l_delay >= 0:
                output[frame_no, 0] += l_attn * src.signal[frame_no-l_delay] / len(sources)
            if frame_no-r_delay >= 0:
                output[frame_no, 1] += r_attn * src.signal[frame_no-r_delay] / len(sources)


class Convolvotron(object):
    def __init__(self, cipic_dir):
        assert os.path.exists(cipic_dir)
        self.hrir_db = CIPIC_HRIR(cipic_dir)

    def binauralize(self, sources):
        if isinstance(sources, PositionalSource):
            sources = [sources]
        num_frames = max([src.signal.shape[0] for src in sources])
        stereo = np.zeros((num_frames, 2))
        for src in sources:
            left_hrir, right_hrir = self.hrir_db.get_closest(src.position)[1:]
            left_signal = np.convolve(src.signal, np.flip(left_hrir, axis=0), 'valid')
            right_signal = np.convolve(src.signal, np.flip(right_hrir, axis=0), 'valid')

            n_valid, i_start = left_signal.shape[0], left_hrir.shape[0] - 1
            stereo[i_start:(i_start + n_valid), 0] += left_signal
            stereo[i_start:(i_start + n_valid), 1] += right_signal
        return stereo

    def binauralize_frame(self, sources, output, frame_no):
        if isinstance(sources, PositionalSource):
            sources = [sources]

        for src in sources:
            left_hrir, right_hrir = self.hrir_db.get_closest(src.position)[1:]

            i_start = frame_no - left_hrir.size + 1 if frame_no >= left_hrir.size else 0
            i_end = frame_no + 1
            i_range = i_end - i_start

            output[frame_no, 0] = (src.signal[i_start:i_end] * left_hrir[-i_range:]).sum()
            output[frame_no, 1] = (src.signal[i_start:i_end] * right_hrir[-i_range:]).sum()


class SourceBinauralizer(object):
    def __init__(self, use_hrtfs=True, cipic_dir=None):
        self.use_hrts = use_hrtfs
        if use_hrtfs:
            self.convolvotron = Convolvotron(cipic_dir)
        else:
            self.stereo_mic = VirtualStereoMic()

    def binauralize(self, sources):
        if isinstance(sources, PositionalSource):
            sources = [sources]
        assert isinstance(sources, list) and all([isinstance(src, PositionalSource) for src in sources])
        assert all([src.sample_rate == sources[0].sample_rate for src in sources])

        if self.use_hrts:
            return self.convolvotron.binauralize(sources)
        else:
            return self.stereo_mic.binauralize(sources)

    def binauralize_frame(self, sources, output, frame_no):
        if isinstance(sources, PositionalSource):
            sources = [sources]
        assert isinstance(sources, list) and all([isinstance(src, PositionalSource) for src in sources])
        assert all([src.sample_rate == sources[0].sample_rate for src in sources])

        if self.use_hrts:
            return self.convolvotron.binauralize_frame(sources, output, frame_no)
        else:
            return self.stereo_mic.binauralize_frame(sources, output, frame_no)


class AmbisonicBinauralizer(object):
    def __init__(self, ambi_format, method='projection', use_hrtfs=False, cipic_dir=None):
        self.source_bin = SourceBinauralizer(cipic_dir=cipic_dir, use_hrtfs=use_hrtfs)
        self.fmt = ambi_format
        self.method = method

        # Initialize speakers
        if self.method == 'pseudoinv':
            self.speaker_pos = map(lambda x: Position(x[0], x[1], x[2], 'cartesian'), get_tDesign(self.fmt.order))
            map(lambda p: p.set_radius(self.fmt.radius), self.speaker_pos)
            # speakers_phi = (2. * np.arange(2*self.fmt.num_channels) / float(2*self.fmt.num_channels) - 1.) * np.pi
            # self.speaker_pos = map(lambda x: Position(x, 0, self.fmt.radius, 'polar'), speakers_phi)
        elif self.method == 'projection':
            speakers_phi = (2. * np.arange(2*self.fmt.num_channels) / float(2*self.fmt.num_channels) - 1.) * np.pi
            self.speaker_pos = map(lambda x: Position(x, 0, self.fmt.radius, 'polar'), speakers_phi)
        else:
            raise ValueError('Unknown decoding method. Options: projection and pseudoinv')
        self.n_speakers = len(self.speaker_pos)
        self.ambi_decoder = AmbiDecoder(self.speaker_pos, self.fmt, method=self.method)

    def binauralize(self, ambi):
        # Decode ambisonics into speakers
        speakers = self.ambi_decoder.decode(ambi)

        # Binauralize speaker as if they were point sources
        sources = map(lambda i: PositionalSource(speakers[:, i], self.speaker_pos[i], self.fmt.sample_rate),
                      range(self.n_speakers))
        stereo = self.source_bin.binauralize(sources)

        return stereo


class DirectAmbisonicBinauralizer(object):
    def __init__(self, ambi_format, method='projection'):
        self.fmt = ambi_format
        self.method = method

        # Initialize ear position
        self.ear_pos = [Position(0, 0.1, 0, 'cartesian'), Position(0, -0.1, 0, 'cartesian')]
        self.ambi_decoder = AmbiDecoder(self.ear_pos, self.fmt, method=self.method)

    def binauralize(self, ambi):
        return self.ambi_decoder.decode(ambi)


def test_convolvotron():
    from pyutils.iolib.audio import load_wav, save_wav
    convolvotron = Convolvotron('hrtfs/cipic_subj3')
    mono, rate = load_wav('wav_test/piano.wav')
    mono = mono[:, 0]

    positions = [[float(num) for num in l.strip().split()] for l in open('wav_test/piano_stat_position.txt', 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]
    source = PositionalSource(mono, positions[0], rate)

    stereo = convolvotron.binauralize([source])
    save_wav('/tmp/output.wav', stereo, rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')

    positions = [[float(num) for num in l.strip().split()] for l in open('wav_test/piano_mov_position.txt', 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]
    source = MovingSource(mono, positions, rate)

    stereo = np.zeros((mono.shape[0], 2))
    while source.tic():
        convolvotron.binauralize_frame([source], stereo, source.cur_idx)
    save_wav('/tmp/output.wav', stereo, rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')


def test_virtual_mic():
    from pyutils.iolib.audio import load_wav, save_wav
    mic = VirtualStereoMic()
    mono, rate = load_wav('wav_test/piano.wav')
    mono = mono[:, 0]

    positions = [[float(num) for num in l.strip().split()] for l in open('wav_test/piano_stat_position.txt', 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]
    source = PositionalSource(mono, positions[0], rate)

    stereo = mic.binauralize([source])
    save_wav('/tmp/output.wav', stereo, rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')

    positions = [[float(num) for num in l.strip().split()] for l in open('wav_test/piano_mov_position.txt', 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]
    source = MovingSource(mono, positions, rate)

    stereo = np.zeros((mono.shape[0], 2))
    while source.tic():
        mic.binauralize_frame([source], stereo, source.cur_idx)
    save_wav('/tmp/output.wav', stereo, rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')


def test_source_binauralizer():
    from pyutils.iolib.audio import load_wav, save_wav
    from pyutils.iolib.position import read_position_file

    # binauralizer = SourceBinauralizer(use_hrtfs=True, cipic_dir='hrtfs/cipic_subj3')
    binauralizer = SourceBinauralizer(use_hrtfs=False)

    # Static source
    sample = 'wav_test/gen_synthetic-S1'
    positions, wav_fns, _, sample_ids = read_position_file(sample+'-position.txt')
    mono, rate = load_wav(wav_fns[sample_ids[0]])
    source = PositionalSource(mono[:, 0], positions[sample_ids[0]][0], rate)
    stereo = binauralizer.binauralize([source])

    save_wav('/tmp/output.wav', stereo / np.abs(stereo).max(), rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')

    # Moving source
    sample = 'wav_test/gen_synthetic-M1'
    positions, wav_fns, _, sample_ids = read_position_file(sample+'-position.txt')
    mono, rate = load_wav(wav_fns[sample_ids[0]])
    source = MovingSource(mono[:, 0], positions[sample_ids[0]], rate)
    stereo = np.zeros((mono.shape[0], 2))
    while source.tic():
        binauralizer.binauralize_frame([source], stereo, source.cur_idx)

    save_wav('/tmp/output.wav', stereo / np.abs(stereo).max(), rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')


def test_ambisonics_binauralizer():
    from pyutils.iolib.audio import load_wav, save_wav
    from pyutils.ambisonics.common import AmbiFormat

    sample = 'wav_test/gen_synthetic-S1'
    ambi, rate = load_wav(sample+'-ambix.wav')

    fmt = AmbiFormat(1, rate)
    binauralizer = DirectAmbisonicBinauralizer(fmt, method='pseudoinv')
    # binauralizer = AmbisonicBinauralizer(fmt, method='projection', use_hrtfs=True, cipic_dir='hrtfs/cipic_subj3')

    stereo = binauralizer.binauralize(ambi)

    save_wav('/tmp/output.wav', stereo / np.abs(stereo).max(), rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')

    sample = 'wav_test/gen_synthetic-M1'
    ambi, rate = load_wav(sample+'-ambix.wav')
    stereo = binauralizer.binauralize(ambi)

    save_wav('/tmp/output.wav', stereo / np.abs(stereo).max(), rate)
    os.system('play /tmp/output.wav')
    os.remove('/tmp/output.wav')


if __name__ == '__main__':
    # test_ambisonics_binauralizer()
    test_source_binauralizer()
    # test_convolvotron()
    # test_virtual_mic()
