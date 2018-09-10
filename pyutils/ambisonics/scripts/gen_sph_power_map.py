import sys, argparse
import numpy as np
from pyutils.ambisonics.distance import SphericalAmbisonicsVisualizer, SphericalSourceVisualizer
from pyutils.iolib.audio import load_wav
from pyutils.iolib.video import VideoWriter
from matplotlib import pyplot as plt


def run(input_fn, output_fn, position_fn='', angular_res=''):
    data, rate = load_wav(input_fn)
    duration = data.shape[0] / float(rate)

    ambiVis = SphericalAmbisonicsVisualizer(data, rate, angular_res=angular_res)
    if position_fn:
        srcVis = SphericalSourceVisualizer(position_fn, duration, ambiVis.visualization_rate(), angular_res=angular_res)

    writer = VideoWriter(output_fn,
                         video_fps=ambiVis.visualization_rate(),
                         width=ambiVis.frame_dims[1],
                         height=ambiVis.frame_dims[0],
                         rgb=True)

    cmap = np.stack(plt.get_cmap('inferno').colors)
    while True:
        frame = ambiVis.get_next_frame()
        if frame is None:
            break
        frame /= frame.max()

        # Super-impose gt position
        if position_fn:
            frame += srcVis.get_next_frame()

        # Process frame and write to disk
        frame = ((frame / frame.max()) * 255).astype(np.uint8)
        frame = (cmap[frame] * 255).astype(np.uint8)    # Add colormap
        writer.write_frame(frame)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_fn',    help='Input ambisonics filename.')
    parser.add_argument('output_fn',   help='Output video filename to store spherical power map.')
    parser.add_argument('--position_fn', default='', help='Ground-truth position file. Source locations will be superimposed.')
    parser.add_argument('--angular_res', default=10., type=float, help='Angular resolution.')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))
