import sys, argparse
import warnings

import numpy as np
from pyutils.ambisonics.binauralizer import SourceBinauralizer
from pyutils.ambisonics.position import PositionalSource, Position, MovingSource
from pyutils.iolib.audio import load_wav, save_wav


def run(input_fn, position_fn, output_fn, use_hrtfs, hrtf_dir):
    mono, rate = load_wav(input_fn)
    if mono.ndim == 2 and mono.shape[1] > 1:
        warnings.warn('Input waveform is not a mono source. Using only first channel.')
        mono = mono[:, 0]

    positions = [[float(num) for num in l.strip().split()] for l in open(position_fn, 'r')]
    positions = [Position(p[0], p[1], p[2], 'polar') for p in positions]
    binauralizer = SourceBinauralizer(use_hrtfs=use_hrtfs, cipic_dir=hrtf_dir)

    if len(positions) == 1:     # Stationary source
        source = PositionalSource(mono, positions[0], rate)
        stereo = binauralizer.binauralize(source)

    else:
        source = MovingSource(mono, positions, rate)
        stereo = np.zeros((mono.shape[0], 2))
        while source.tic():
            binauralizer.binauralize_frame(source, stereo, source.cur_idx)

    save_wav(output_fn, stereo, rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_fn',  help='Input mono file.')
    parser.add_argument('position_fn',  help='Input mono file.')
    parser.add_argument('output_fn',  help='Output file.')
    parser.add_argument('--use_hrtfs', action='store_true',  help='Whether to use hrtfs.')
    parser.add_argument('--hrtf_dir', default='', help='Input hrtf directory.')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))
