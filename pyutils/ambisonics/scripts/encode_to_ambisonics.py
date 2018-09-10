import argparse
import sys
import os
import librosa

import numpy as np
from pyutils.ambisonics.common import AmbiFormat, AmbisonicArray
from pyutils.ambisonics.encoder import AmbiEncoder
from pyutils.ambisonics.position import MovingSource
from pyutils.iolib.audio import save_wav
from pyutils.iolib.position import read_position_file
ESC_BASE = '/mnt/ilcompf9d1/user/morgado/spatialaudiogen/data/ESC-50'

def run(position_fn, ambi_order, output_fn, rate=24000, base_dir=None, randomize=False, overwrite=False):
    if overwrite and os.path.exists(output_fn):
        os.remove(output_fn)
    assert not os.path.exists(output_fn)

    if base_dir is None:
        base_dir = ESC_BASE

    sample_ids, positions, input_fn, _, _ = read_position_file(position_fn)
    source, _ = librosa.load(os.path.join(base_dir, input_fn['source']), sr=rate)
    bkg, _ = librosa.load(os.path.join(base_dir, input_fn['ambient']), sr=rate)
    Psrc = np.convolve(source ** 2, np.ones((int(rate * 0.1),)) / (rate * 0.1)).max()
    Pbkg = np.convolve(bkg ** 2, np.ones((int(rate * 0.1),)) / (rate * 0.1)).max()
    bkg *= 0.1*Psrc/Pbkg

    data = {}
    for smp_id in sample_ids:
        fn = os.path.join(base_dir, input_fn[smp_id])
        mono, _ = librosa.load(fn, sr=rate)
        if mono.ndim == 2:
            mono = mono[:, 0]
        data[smp_id] = mono

    fmt = AmbiFormat(ambi_order=ambi_order, sample_rate=rate)
    encoder = AmbiEncoder(fmt)

    sources = [MovingSource(data[smp_id], positions[smp_id], rate)
               for smp_id in sample_ids if len(positions[smp_id])]
    nframes = max([v.shape[0] for v in data.values()])
    ambix = AmbisonicArray(np.zeros((nframes, fmt.num_channels)), fmt)
    t = -1
    while all([src.tic() for src in sources]):
        t += 1
        encoder.encode_frame(sources, ambix, t)

    ambix = ambix.data
    for smp_id in sample_ids:
        if len(positions[smp_id]) == 0: # Ambient sound
            ambix[:data[smp_id].size, 0] += data[smp_id]
    ambix = ambix / ambix.max() * 0.95

    save_wav(output_fn, ambix, rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('position_fn',  help='Input position file.')
    parser.add_argument('ambi_order', default=1, type=int, help='Ambisonics order.')
    parser.add_argument('output_fn',  help='Output file.')
    parser.add_argument('--rate', default=24000, type=int, help='Frame rate.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))
