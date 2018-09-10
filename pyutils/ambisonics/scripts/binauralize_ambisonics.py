import sys, os, argparse
import numpy as np
from pyutils.ambisonics.binauralizer import DirectAmbisonicBinauralizer
from pyutils.ambisonics.common import AmbiFormat
from pyutils.iolib.audio import load_wav, save_wav


def run(input_fn, output_fn, overwrite=False):
    if overwrite and os.path.exists(output_fn):
        os.remove(output_fn)
    assert not os.path.exists(output_fn)

    data, rate = load_wav(input_fn)
    ambi_order = int(np.sqrt(data.shape[1]) - 1)

    fmt = AmbiFormat(ambi_order=ambi_order, sample_rate=rate)
    binauralizer = DirectAmbisonicBinauralizer(fmt, method='pseudoinv')
    # binauralizer = AmbisonicBinauralizer(fmt, method='projection', use_hrtfs=use_hrtfs, cipic_dir=hrtf_dir)
    stereo = binauralizer.binauralize(data)
    save_wav(output_fn, stereo, rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_fn',  help='Input ambisonics file.')
    parser.add_argument('output_fn',  help='Output stereo file.')
    parser.add_argument('--overwrite', action='store_true',  help='Whether to overwrite output file.')
    # parser.add_argument('--use_hrtfs', action='store_true',  help='Whether to use hrtfs.')
    # parser.add_argument('--hrtf_dir', default='', help='Input hrtf directory.')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))
