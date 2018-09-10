import argparse
import sys
import warnings

from pyutils.ambisonics.common import AmbiFormat
from pyutils.ambisonics.encoder import AmbiEncoder
from pyutils.ambisonics.position import PositionalSource, Position
from pyutils.iolib.audio import load_wav, save_wav


def run(input_fn, x, y, z, ambi_order, output_fn):
    mono, rate = load_wav(input_fn)
    if mono.ndim == 2 and mono.shape[1] > 1:
        warnings.warn('Input waveform is nor a mono source. Using only first channel.')
        mono = mono[:, 0]

    encoder = AmbiEncoder(AmbiFormat(ambi_order=ambi_order, sample_rate=rate))
    source = PositionalSource(mono, Position(x, y, z, 'cartesian'), rate)
    ambi = encoder.encode(source)
    save_wav(output_fn, ambi.data, rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_fn',  help='Input mono file.')
    parser.add_argument('x', default=1., type=float, help='x coordinate.')
    parser.add_argument('y', default=0., type=float, help='y coordinate.')
    parser.add_argument('z', default=0., type=float, help='z coordinate.')
    parser.add_argument('ambi_order', default=1, type=int, help='Ambisonics order.')
    parser.add_argument('output_fn',  help='Output file.')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))
