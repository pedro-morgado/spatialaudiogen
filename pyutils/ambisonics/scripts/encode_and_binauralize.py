import argparse
import sys
import warnings

import numpy as np
from pyutils.ambisonics.binauralizer import DirectAmbisonicBinauralizer
from pyutils.ambisonics.common import AmbiFormat, AmbisonicArray
from pyutils.ambisonics.encoder import AmbiEncoder
from pyutils.ambisonics.position import PositionalSource, Position, MovingSource
from pyutils.iolib.audio import load_wav, save_wav


def run(input_fn, position_fn, ambi_order, output_fn):
    mono, rate = load_wav(input_fn)
    if mono.ndim == 2 and mono.shape[1] > 1:
        warnings.warn('Input waveform is not a mono source. Using only first channel.')
        mono = mono[:, 0]

    fmt = AmbiFormat(ambi_order=ambi_order, sample_rate=rate)
    encoder = AmbiEncoder(fmt)
    positions = [np.array([float(num) for num in l.strip().split()]) for l in open(position_fn, 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]

    if len(positions) == 1:
        # Stationary source
        source = PositionalSource(mono, positions[0], rate)
        ambi = encoder.encode(source)

    else:
        source = MovingSource(mono, positions, rate)
        ambi = AmbisonicArray(np.zeros((mono.shape[0], fmt.num_channels)), fmt)
        while source.tic():
            encoder.encode_frame(source, ambi, source.cur_idx)

    binauralizer = DirectAmbisonicBinauralizer(fmt, method='projection')
    # binauralizer = AmbisonicBinauralizer(fmt, method='projection', use_hrtfs=use_hrtfs, cipic_dir=hrtf_dir)
    stereo = binauralizer.binauralize(ambi.data)
    save_wav(output_fn, stereo, rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_fn',  help='Input mono file.')
    parser.add_argument('position_fn',  help='Input position file.')
    parser.add_argument('ambi_order', default=1, type=int, help='Ambisonics order.')
    parser.add_argument('output_fn',  help='Output file.')
    # parser.add_argument('--use_hrtfs', action='store_true',  help='Whether to use hrtfs.')
    # parser.add_argument('--hrtf_dir', default='', help='Input hrtf directory.')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))
