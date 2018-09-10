import sys, os
from datetime import datetime
import numpy as np
import tensorflow as tf
from scipy.misc import imresize


def add_scalar_summaries(tensor_list, tensor_names):
    if tensor_list:
        # Attach a scalar summary to all individual losses and metrics.
        for name, tensor in zip(tensor_names, tensor_list):
            tf.summary.scalar('%s' % name, tensor)


def print_stats(values, names, batch_size, step_duration, curr_step, tag='TRAIN'):
    # Print stats to shell
    samples_per_sec = len(step_duration)*batch_size/sum(step_duration)
    timestamp = datetime.now()

    stats_str = '%s: %s | step %d' % (timestamp, tag, curr_step)
    stats_str += ' (%.3f samples/sec | %.3f secs/sample)' % (samples_per_sec, 1./samples_per_sec)
    metrics_str = '\n'.join(['%s: %s | \t %s %f' % (timestamp, tag, name, l) for l, name in zip(values, names)])

    print(stats_str)
    print(metrics_str)
    sys.stdout.flush()


def save_params(args):
    with open(args.model_dir+'/train-params.txt', 'w') as f:
        for k, v in args.__dict__.iteritems():
            f.write('{}: {}\n'.format(k, v))


def print_params(args):
    for k, v in args.__dict__.iteritems():
        print '{}: {}'.format(k, v)


def load_params(model_dir):
    params = {l.split(':')[0]: l.strip().split(':')[1].strip()
              for l in open(model_dir+'/train-params.txt')}
    for k in ['encoders', 'separation']:
        params[k] = params[k].lower()
    params['ambi_order'] = int(params['ambi_order'])
    params['audio_rate'] = int(params['audio_rate'])
    params['video_rate'] = int(params['video_rate'])
    params['context'] = float(params['context'])
    params['sample_dur'] = float(params['sample_dur'])
    params['encoders'] = [enc.strip()[1:-1] for enc in params['encoders'][1:-1].split(',')]
    params['lr'] = float(params['lr'])
    params['n_iters'] = int(params['n_iters'])
    params['batch_size'] = int(params['batch_size'])
    params['lr_decay'] = float(params['lr_decay'])
    params['lr_iters'] = float(params['lr_iters'])
    if 'num_sep_tracks' not in params:
        params['num_sep_tracks'] = '64'
    params['num_sep_tracks'] = int(params['num_sep_tracks'])
    if 'fft_window' not in params:
        params['fft_window'] = '0.025'
    params['fft_window'] = float(params['fft_window'])
    if 'context_units' not in params:
        params['context_units'] = '[64, 128, 128]'
    if len(params['context_units'][1:-1]) > 0:
        params['context_units'] = [int(l.strip()) for l in params['context_units'][1:-1].split(',')]
    else:
        params['context_units'] = []
    if 'freq_mask_units' not in params:
        params['freq_mask_units'] = '[]'
    if len(params['freq_mask_units'][1:-1]) > 0:
        params['freq_mask_units'] = [int(l.strip()) for l in params['freq_mask_units'][1:-1].split(',')]
    else:
        params['freq_mask_units'] = []
    if 'loc_units' not in params:
        params['loc_units'] = '[256, 256]'
    if len(params['loc_units'][1:-1]) > 0:
        params['loc_units'] = [int(l.strip()) for l in params['loc_units'][1:-1].split(',')]
    else:
        params['loc_units'] = []

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Struct(**params)


def img_prep_fcn():
    return lambda x: x/255. - 0.5


def flow_prep_fcn():
    return lambda x: imresize(x, (224, 448), 'nearest')


def compute_lsd_dist(pred, gt, rate):
    import librosa
    def power_spect(x):
        EPS = 1e-2
        return 10 * np.log(np.abs(x)+EPS) / np.log(10.)
    dist = np.zeros(gt.shape[1])
    for i in range(gt.shape[1]):
        spect_pred = librosa.feature.melspectrogram(y=pred[:, i], sr=rate, n_mels=128, fmax=12000)
        spect_gt = librosa.feature.melspectrogram(y=gt[:, i], sr=rate, n_mels=128, fmax=12000)
        dist[i] = np.sqrt(np.mean((power_spect(spect_gt) - power_spect(spect_pred))**2))
    return dist


def compute_envelope_dist(pred, gt):
    from scipy.signal import hilbert
    dist = np.zeros(gt.shape[1])
    for i in range(gt.shape[1]):
        pred_env = np.abs(hilbert(pred[:, i]))
        gt_env = np.abs(hilbert(gt[:, i]))
        dist[i] = np.sqrt(np.mean((gt_env - pred_env)**2))
    return dist


def stft(inp, wind_size, n_overlap):
    inp_sz = inp.get_shape().as_list()
    if len(inp_sz) > 2:
        inp = tf.reshape(inp, (np.prod(inp_sz[:-1]), inp_sz[-1]))

    batch_size = inp.get_shape().as_list()[0]
    n_frames = inp.get_shape().as_list()[-1]
    n_winds = int(np.floor(n_frames / wind_size)) - 1
    x_crops = []
    for k, ss in enumerate(range(0, wind_size, wind_size / n_overlap)):
        x_crops.append(inp[:, ss:ss + wind_size * n_winds])

    x = tf.stack(x_crops, 1)
    x = tf.reshape(x, (batch_size, n_overlap, -1, wind_size))

    hann_window = tf.constant(0.5 - (0.5 * np.cos(2 * np.pi / wind_size * np.arange(wind_size))), dtype=tf.float32)
    hann_window = tf.expand_dims(tf.expand_dims(hann_window, 0), 0)
    x = x * hann_window

    stft = tf.fft(tf.cast(x, tf.complex64))
    stft = tf.transpose(stft, (0, 2, 1, 3))

    sz = stft.get_shape().as_list()
    stft = tf.reshape(stft, (sz[0], sz[1] * sz[2], sz[3]))

    if len(inp_sz) > 2:
        stft_sz = stft.get_shape().as_list()
        stft = tf.reshape(stft, inp_sz[:-1]+stft_sz[-2:])
    return stft



def stft_for_loss(signal, window, n_overlap):
    BS, N, nC = signal.get_shape().as_list()

    # FFT on 2**n windows is faster
    window = int(2**np.ceil(np.log(window)/np.log(2)))
    hann_window = 0.5 - (0.5 * np.cos(2 * np.pi / window * np.arange(window)))

    if n_overlap == 1:
        nW = int(float(N) / window)
        if nW > 1:
            if N > window * nW:
                signal = signal[:, :window * nW, :]
            windows = tf.reshape(signal, (BS, nW, window, nC))
        else:
            windows = signal
    else:
        windows = []
        stride = int(window / n_overlap)  # frames
        for i in range(n_overlap):
            nW = int(float(N - i * stride - 1) / window)
            y = signal[:, (i * stride):(i * stride) + window * nW, :]
            windows.append(tf.reshape(y, (BS, nW, window, nC)))
        windows = tf.concat(windows, 1)

    windows = tf.transpose(windows, (0, 3, 1, 2))
    windows *= hann_window[np.newaxis, np.newaxis, np.newaxis, :]
    fft = tf.fft(tf.cast(windows, tf.complex64))
    return fft


def istft(inp, n_overlap):
    inp_sz = inp.get_shape().as_list()
    if len(inp_sz) > 3:
        inp = tf.reshape(inp, (np.prod(inp_sz[:-2]), inp_sz[-2], inp_sz[-1]))

    batch_size, n_frames, n_freqs = inp.get_shape().as_list()
    n_frames = int(int(float(n_frames)/n_overlap)*n_overlap)
    inp = inp[:, :n_frames, :]
    batch_size, n_frames, n_freqs = inp.get_shape().as_list()

    x = tf.real(tf.ifft(inp))
    x = tf.reshape(x, (batch_size, -1, n_overlap, n_freqs))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (batch_size, n_overlap, -1))

    x_list = tf.unstack(x, axis=1)
    skip = n_freqs / n_overlap
    for i in range(n_overlap):
        # x_sep[i] = tf.manip.roll(x_sep[i], i*wind_size/4, 2)
        if i == 0:
            x_list[i] = x_list[i][:, (n_overlap - i - 1)*skip:]
        else:
            x_list[i] = x_list[i][:, (n_overlap - i - 1)*skip:-i*skip]
        
    x = tf.add_n(x_list) / float(n_overlap)

    if len(inp_sz) > 3:
        x_sz = x.get_shape().as_list()
        x = tf.reshape(x, inp_sz[:-2]+x_sz[-1:])

    return x


def optimize(total_loss, step_t, params):
    lr_t = tf.train.exponential_decay(params.lr, step_t,
                                      decay_steps=params.lr_iters,
                                      decay_rate=params.lr_decay,
                                      staircase=True)

    opt = tf.train.AdamOptimizer(lr_t)
    train_op = opt.minimize(total_loss, global_step=step_t)
    return train_op, lr_t


def gen_360video(audio_fn, video_fn, output_fn, inject_meta=False, overlay_map=False, binauralize=False, no_spatial_audio=False):
    from pyutils.iolib.video import VideoReader, VideoWriter
    from pyutils.iolib.audio import load_wav, save_wav
    from pyutils.ambisonics.distance import SphericalAmbisonicsVisualizer
    import tempfile
    from matplotlib import pyplot as plt
    from skimage.transform import resize

    tmp_file = tempfile.mktemp(dir='/tmp/', suffix='.mp4')
    tmp_snd_file = tempfile.mktemp(dir='/tmp/', suffix='.wav')
    tmp_vid_file = tempfile.mktemp(dir='/tmp/', suffix='.mp4')
    
    print('Splitting')
    cmd = 'ffmpeg -i {} -vn -strict -2 {}'.format(audio_fn, tmp_snd_file)
    print(cmd)
    os.system(cmd)

    cmd = 'ffmpeg -i {} -an -vcodec copy {}'.format(video_fn, tmp_vid_file) 
    print(cmd)
    os.system(cmd)

    if overlay_map:
        print('Overlaying spherical map')
        tmp_vid_file2 = tempfile.mktemp(dir='/tmp/', suffix='.mp4')
        ambix, snd_rate = load_wav(tmp_snd_file)
        reader = VideoReader(tmp_vid_file, rate=10)
        writer = VideoWriter(tmp_vid_file2, reader.fps)
        ambiVis = SphericalAmbisonicsVisualizer(ambix[::5], snd_rate/5., 5./reader.fps, 5.)
        cmap = plt.cm.YlOrRd(np.linspace(0, 1, 256))[:, :3]

        cur_rms = ambiVis.get_next_frame()
        cur_rms = (cur_rms - cur_rms.min()) / (cur_rms.max() - cur_rms.min() + 0.005)
        while True:
            prev_rms = cur_rms
            cur_rms = ambiVis.get_next_frame()
            if cur_rms is None:
                break
            cur_rms = (cur_rms - cur_rms.min()) / (cur_rms.max() - cur_rms.min() + 0.005)

            for i in range(5):
                frame = reader.get()
                if frame is None:
                    break

                beta = i/5.
                rms = (1 - beta) * prev_rms + beta * cur_rms
                rms = rms*2. - 0.7
                rms[rms<0] = 0
                dir_map = (rms * 255).astype(int)
                dir_map[dir_map > 255] = 255
                dir_map = resize(cmap[dir_map], reader.frame_shape[:2]) * 255

                alpha = resize(rms[:, :, np.newaxis], reader.frame_shape[:2]) * 0.6
                overlay = alpha * dir_map + (1 - alpha) * frame
                writer.write_frame(overlay.astype(np.uint8))

        del writer, reader
        os.remove(tmp_vid_file)
        tmp_vid_file = tmp_vid_file2

    if binauralize:
        print('Binauralizing')
        tmp_snd_file2 = tempfile.mktemp(dir='/tmp/', suffix='.wav')
        ambix, snd_rate = load_wav(tmp_snd_file)
        stereo = np.stack([ambix[:,0]+ambix[:,1], ambix[:,0]-ambix[:,1]], 1)
        stereo /= (np.abs(stereo).max() / 0.95)
        save_wav(tmp_snd_file2, stereo, snd_rate)

        os.remove(tmp_snd_file)
        tmp_snd_file = tmp_snd_file2

    print('Mixing')
    cmd = 'ffmpeg -y -i {} -i {} -vcodec copy -strict -2 {}'.format(tmp_snd_file, tmp_vid_file, tmp_file)
    print(cmd)
    os.system(cmd)

    cwd = os.getcwd()
    output_fn = os.path.join(cwd, output_fn)

    if inject_meta:
        print('Injecting metadata')
        file_dir = os.path.dirname(os.path.realpath(__file__))
        spt_media_dir = os.path.realpath(os.path.join(file_dir, '..', '3rd-party', 'spatial-media'))
        os.chdir(spt_media_dir)
        os.system('python spatialmedia -i --stereo=none {} {} {} '.format('' if no_spatial_audio else '--spatial-audio', tmp_file, output_fn))
        os.chdir(cwd)
        os.remove(tmp_file)

    else:
        import shutil
        shutil.move(tmp_file, output_fn)

    os.remove(tmp_snd_file)
    os.remove(tmp_vid_file)