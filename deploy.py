"""Description
"""
import sys, os, tempfile, argparse
import tensorflow as tf
import numpy as np

from definitions import *
from feeder import SampleReader
from model import SptAudioGen, SptAudioGenParams
from pyutils.iolib.audio import save_wav
import myutils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_dir', help='Directory containing model snapshot.')

    # Inputs
    parser.add_argument('input_folder', default='', help='Folder with input sample.')
    parser.add_argument('video', default='', help='High resolution video.')
    parser.add_argument('--deploy_start', default=0., type=float)
    parser.add_argument('--deploy_duration', default=10., type=float)

    # Outputs
    parser.add_argument('--output_fn', default='output', help='Basename for output files.')
    parser.add_argument('--save_ambix', action='store_true', help='Output ambix video file.')
    parser.add_argument('--save_video', action='store_true', help='Output video file.')
    parser.add_argument('--overlay_map', action='store_true', help='Overlay spherical map.')
    parser.add_argument('--VR', action='store_true', help='360 video.')

    parser.add_argument('--gpu', type=int, default=0, help="GPU id")

    args = parser.parse_args(sys.argv[1:])
    if args.deploy_duration <= 0:
        args.deploy_duration = None
    return args


class W2XYZ(object):
    def __init__(self, model_dir):
        print('\n' + '='*30 + ' ARGUMENTS ' + '='*30)
        params = myutils.load_params(model_dir)
        for k, v in params.__dict__.iteritems():
            print('TRAIN  | {}: {}'.format(k, v))

        self.params = params
        self.duration = 0.1
        self.batch_size = 10
        min_t = min([params.context, self.duration, 1. / params.video_rate])

        # Model
        num_sep = params.num_sep_tracks if params.separation != NO_SEPARATION else 1
        net_params = SptAudioGenParams(sep_num_tracks=num_sep, ctx_feats_fc_units=params.context_units,
                                       loc_fc_units=params.loc_units, sep_freq_mask_fc_units=params.freq_mask_units,
                                       sep_fft_window=params.fft_window)
        self.model = SptAudioGen(ambi_order=params.ambi_order, 
                                 audio_rate=params.audio_rate,
                                 video_rate=params.video_rate, 
                                 context=params.context,
                                 sample_duration=self.duration, 
                                 encoders=params.encoders,
                                 separation=params.separation,
                                 params=net_params)
        
        self.audio_size = self.model.snd_dur + self.model.snd_contx - 1
        self.video_size = int(self.duration * params.video_rate)
        shape = (self.batch_size, self.audio_size, 1)
        self.tba = {AUDIO: tf.placeholder(dtype=tf.float32, shape=shape)}
        if VIDEO in params.encoders:
            shape = (self.batch_size, self.video_size, 224, 448, 3)
            self.tba[VIDEO] = tf.placeholder(dtype=tf.float32, shape=shape)
        if FLOW in params.encoders:
            shape = (self.batch_size, self.video_size, 224, 448, 3)
            self.tba[FLOW] = tf.placeholder(dtype=tf.float32, shape=shape)
        self.ambi_pred_t = self.model.inference_ops(is_training=False, **self.tba)
        
        saver = tf.train.Saver()
        config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        self.sess = tf.Session(config=config)
        print('Loading model...')
        print(tf.train.latest_checkpoint(model_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        #saver.restore(self.sess, model_dir+'/model.ckpt')

    def deploy(self, input_folder, deploy_start, deploy_duration):
        reader = SampleReader(input_folder,
                              ambi_order=self.params.ambi_order,
                              audio_rate=self.params.audio_rate,
                              video_rate=self.params.video_rate,
                              context=self.params.context,
                              duration=self.duration,
                              return_video=VIDEO in self.params.encoders,
                              img_prep=myutils.img_prep_fcn(),
                              return_flow=FLOW in self.params.encoders,
                              start_time=deploy_start,
                              sample_duration=deploy_duration,
                              skip_silence_thr=None,
                              shuffle=False,
                              random_rotations=False,
                              skip_rate=None)
        dt = reader.chunks_t[0] - deploy_start
        reader.chunks_t = [t - dt for t in reader.chunks_t]

        print('Generating ambisonics...')
        ss = self.model.snd_contx / 2
        mono, ambi_pred, sep_channels, sep_mask, weights, biases = [], [], [], [], [], []
        while True:
            batch = []
            for _ in range(self.batch_size):
                chunk = reader.get()
                if chunk is None:
                    break
                batch.append(chunk)
            if not batch:
                break
            
            vids = [b['id'] for b in batch]
            n_samples = len(vids)
            ambix = np.stack([b['ambix'] for b in batch], axis=0)
            if n_samples != self.batch_size:
                ambix = np.concatenate([ambix, np.zeros((self.batch_size - n_samples, ambix.shape[1], ambix.shape[2]))], axis=0)
            feed_dict = {self.tba[AUDIO]: ambix[:, :, :1]}

            if VIDEO in self.params.encoders:
                video = np.stack([b['video'] for b in batch], axis=0)
                if n_samples != self.batch_size:
                    video = np.concatenate([video, np.zeros((self.batch_size - n_samples, video.shape[1], video.shape[2], video.shape[3], video.shape[4]))], axis=0)
                feed_dict[self.tba[VIDEO]] = video

            if FLOW in self.params.encoders:
                flow = np.stack([b['flow'] for b in batch], axis=0)
                if n_samples != self.batch_size:
                    flow = np.concatenate([flow, np.zeros((self.batch_size - n_samples, flow.shape[1], flow.shape[2], flow.shape[3], flow.shape[4]))], axis=0)
                feed_dict[self.tba[FLOW]] = flow

            ambi_pred_chk = self.sess.run(self.ambi_pred_t, feed_dict=feed_dict)

            n_frames = n_samples * ambi_pred_chk.shape[1]
            n_out = ambi_pred_chk.shape[2]

            ambi_pred_chk = np.copy(ambi_pred_chk[:n_samples]).reshape((n_frames, n_out))
            ambi_pred.append(ambi_pred_chk)
            mono.append(np.copy(ambix[:n_samples, ss:ss + self.model.snd_dur, :1]).reshape((-1, 1)))

        mono = np.concatenate(mono, 0)
        ambi_pred = np.concatenate((mono, np.concatenate(ambi_pred, 0)), 1)
        return ambi_pred


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    tmp_ambix_fn = tempfile.mktemp(prefix='/tmp/', suffix='.wav')
    tmp_video_fn = tempfile.mktemp(prefix='/tmp/', suffix='.mp4')

    model = W2XYZ(args.model_dir)
    ambi_pred = model.deploy(args.input_folder, args.deploy_start, args.deploy_duration)

    # dur_t = model.model.duration
    # snd1 = model.deploy(args.input_folder, args.deploy_start - dur_t/2, args.deploy_duration + dur_t)    
    # hann1 = np.hanning(model.model.snd_dur)
    # hann1 = np.tile(hann1, snd1.shape[0]/hann1.size)[:, np.newaxis]
    # ss = model.model.snd_dur/2
    # t = int(args.deploy_duration * model.params.audio_rate)
    # snd1 = snd1[ss:ss+t]
    # hann1 = hann1[ss:ss+t]

    # snd2 = model.deploy(args.input_folder, args.deploy_start, args.deploy_duration + dur_t)
    # hann2 = np.hanning(model.model.snd_dur)
    # hann2 = np.tile(hann2, snd2.shape[0]/hann2.size)[:, np.newaxis]
    # ss = 0
    # t = int(args.deploy_duration * model.params.audio_rate)
    # snd2 = snd2[ss:ss+t]
    # hann2 = hann2[ss:ss+t]

    # ambi_pred = (snd1 * hann1 + snd2 * hann2) / (hann1 + hann2)

    # Save ambisonics
    save_wav(tmp_ambix_fn, ambi_pred, model.params.audio_rate)

    if args.save_ambix:
        print('Saving ambisonics wav...')
        cmd = 'ffmpeg -y -i {} -strict -2 {}'.format(tmp_ambix_fn, args.output_fn)
        os.system(cmd)

    if args.save_video:
        print('Saving video...')
        cmd = 'ffmpeg -y -ss {} -i {} -t {} {}'.format(args.deploy_start, args.video, args.deploy_duration, tmp_video_fn)
        os.system(cmd)

        myutils.gen_360video(tmp_ambix_fn, tmp_video_fn, args.output_fn, overlay_map=args.overlay_map, inject_meta=args.VR, binauralize=not args.VR)

        os.remove(tmp_video_fn)
    os.remove(tmp_ambix_fn)

if __name__ == '__main__':
    print(os.getcwd())
    main(parse_arguments())
