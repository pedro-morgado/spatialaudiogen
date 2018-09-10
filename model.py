import numpy as np
import tensorflow as tf
import pyutils.tflib.wrappers as tfw
from pyutils.tflib.models.image.resnet import ResNet18
from collections import OrderedDict
import myutils
from definitions import *


class SptAudioGenParams:
    def __init__(self,
                 sep_num_tracks=NUM_SEP_TRACKS_DEF,
                 ctx_feats_fc_units=CTX_FEATS_FCUNITS_DEF,
                 loc_fc_units=LOC_FCUNITS_DEF,
                 sep_freq_mask_fc_units=SEP_FREQ_MASK_FCUNITS_DEF,
                 sep_fft_window=SEP_FFT_WINDOW_DEF):
        self.sep_num_tracks = sep_num_tracks
        self.ctx_feats_fc_units = ctx_feats_fc_units
        self.loc_fc_units = loc_fc_units
        self.sep_freq_mask_fc_units = sep_freq_mask_fc_units
        self.sep_fft_window = sep_fft_window


class SptAudioGen(object):
    def __init__(self, ambi_order,
                 audio_rate=48000,
                 video_rate=10,
                 context=1.,
                 sample_duration=0.1,
                 encoders=None,
                 separation='none',
                 params=SptAudioGenParams()):
        assert float(audio_rate)/video_rate == int(audio_rate)/int(video_rate)

        self.ambi_order = ambi_order
        self.num_ambi_channels = sum([2*i+1 for i in range(ambi_order+1)])
        self.snd_rate, self.vid_rate = audio_rate, video_rate
        self.context, self.duration = context, sample_duration
        self.snd_contx = int(context * audio_rate)
        self.snd_dur = int(sample_duration * audio_rate)
        self.snd_size = self.snd_contx + self.snd_dur - 1
        assert self.snd_rate % self.vid_rate == 0

        if encoders is None:
            encoders = [AUDIO, VIDEO, FLOW]
        assert isinstance(encoders, list)
        assert all([e in ENCODERS for e in encoders])
        self.encoders = encoders
        self.separation = separation
        self.params = params

        self.model = None
        self.deploy = None
        self.solver = None
        self.ends = OrderedDict()
        self.init_ops = []
        self.loc_channels = None
        self.sep_channels = None
        self.wind_size = int(self.params.sep_fft_window * self.snd_rate)
        self.wind_size = int(2**np.round(np.log2(self.wind_size)))

    @staticmethod
    def _stft_mse_ops(gt, pred, window, overlap):
        with tf.variable_scope('stft_diff'):
            with tf.variable_scope('stft_gt'):
                # stft_gt = myutils.stft(tf.transpose(gt, (0, 2, 1)), window, overlap)
                stft_gt = myutils.stft_for_loss(gt, window, overlap)

            with tf.variable_scope('stft_pred'):
                # stft_pred = myutils.stft(tf.transpose(pred, (0, 2, 1)), window, overlap)
                stft_pred = myutils.stft_for_loss(pred, window, overlap)

            with tf.variable_scope('mse'):
                stft_diff = tf.abs(stft_gt-stft_pred)
                mse = tf.reduce_mean(tf.reduce_mean(stft_diff**2, axis=3), axis=2)
        return mse

    @staticmethod
    def _lsd_ops(gt, pred, window, overlap):
        EPS = 1e-2
        with tf.variable_scope('lsd'):
            with tf.variable_scope('stft_gt'):
                stft_gt = myutils.stft(tf.transpose(gt, (0, 2, 1)), window, overlap)

            with tf.variable_scope('stft_pred'):
                stft_pred = myutils.stft(tf.transpose(pred, (0, 2, 1)), window, overlap)

            with tf.variable_scope('lsd'):
                def power_spect(x):
                    return 10 * tf.log(tf.abs(x)+EPS) / tf.log(10.)
                log_spec_diff = (power_spect(stft_gt) - power_spect(stft_pred))
                lsd_t = tf.sqrt(tf.reduce_mean(log_spec_diff**2, axis=3))
                lsd = tf.reduce_mean(lsd_t, axis=2)
                return lsd

    @staticmethod
    def _temporal_mse_ops(gt, pred):
        with tf.variable_scope('mse'):
            return tf.reduce_mean((gt - pred)**2, axis=1)

    @staticmethod
    def _temporal_snr_ops(gt, pred):
        EPS = 1e-1
        with tf.variable_scope('snr'):
            Psignal = tf.reduce_sum(gt**2, axis=1)
            Pnoise = tf.reduce_sum((gt-pred)**2, axis=1)
            snr = 10. * tf.log((Psignal+EPS)/(Pnoise+EPS)) / tf.log(10.)
            return snr

    def evaluation_ops(self, preds_t, targets_t, w_t, mask_channels=None):
        print('\n Metrics')
        print(' * {:15s} | {:20s} | {:10s}'.format('Prediction', str(preds_t.get_shape()), str(preds_t.dtype)))
        print(' * {:15s} | {:20s} | {:10s}'.format('Target', str(targets_t.get_shape()), str(targets_t.dtype)))
        print(' * {:15s} | {:20s} | {:10s}'.format('Channel mask', str(mask_channels.get_shape()), str(mask_channels.dtype)))

        if mask_channels is None:
            batch_size, _, n_channels = preds_t.get_shape()
            mask_channels = tf.ones((batch_size, n_channels))
        num_masked = tf.reduce_sum(mask_channels, axis=0)
        num_masked = tf.maximum(num_masked, 1)

        metrics = OrderedDict()
        window = int(FFT_WINDOW * self.snd_rate)
        overlap = FFT_OVERLAP_R
        stft_dist_ps = self._stft_mse_ops(targets_t, preds_t, window, overlap)
        stft_dist = tf.reduce_sum(stft_dist_ps * mask_channels, axis=0) / num_masked * 100.
        metrics['stft/avg'] = tf.reduce_mean(stft_dist)
        for i, ch in zip(range(3), 'YZX'):
            metrics['stft/'+ch] = stft_dist[i]

        lsd_ps = self._lsd_ops(targets_t, preds_t, window, overlap)
        lsd = tf.reduce_sum(lsd_ps * mask_channels, axis=0) / num_masked
        metrics['lsd/avg'] = tf.reduce_mean(lsd)
        for i, ch in zip(range(3), 'YZX'):
            metrics['lsd/'+ch] = lsd[i]

        mse_ps = self._temporal_mse_ops(targets_t, preds_t)
        mse = tf.reduce_sum(mse_ps * mask_channels, axis=0) / num_masked * 5e3
        metrics['mse/avg'] = tf.reduce_mean(mse)
        for i, ch in zip(range(3), 'YZX'):
            metrics['mse/'+ch] = mse[i]

        snr_ps = self._temporal_snr_ops(targets_t, preds_t)
        snr = tf.reduce_sum(snr_ps * mask_channels, axis=0) / num_masked
        metrics['snr/avg'] = tf.reduce_mean(snr)
        for i, ch in zip(range(3), 'YZX'):
            metrics['snr/'+ch] = snr[i]

        metrics['pow/pred'] = tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(preds_t ** 2, axis=2), axis=0))
        metrics['pow/gt'] = tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(targets_t ** 2, axis=2), axis=0))

        for m in metrics:
            print(' * {:15s} | {:20s} | {:10s}'.format(m, str(metrics[m].get_shape()), str(metrics[m].dtype)))
        return metrics, stft_dist_ps, lsd_ps, mse_ps, snr_ps

    def loss_ops(self, metrics_t, step_t):
        losses = OrderedDict()
        losses['stft/mse'] = metrics_t['stft/avg']
        return losses

    def audio_encoder_ops(self, stft):
        n_filters = [32, 64, 128, 256, 512]
        filter_size = [(7, 16), (3, 7), (3, 5), (3, 5), (3, 5)]
        stride = [(4, 8), (2, 4), (2, 2), (1, 1), (1, 1)]

        inp_dim = 95.    # Encoder Dim=1
        ss = (self.snd_contx / 2.) * (4. / self.wind_size)
        ss = int(ss - (inp_dim - 1) / 2.)

        tt = (self.snd_contx / 2. + self.snd_dur) * (4. / self.wind_size)
        tt = int(tt + (inp_dim - 1) / 2.)
        tt = int((np.ceil((tt - ss - inp_dim) / 16.)) * 16 + inp_dim + ss)

        sz = stft.get_shape().as_list()
        stft = tf.transpose(stft[:, :, ss:tt, :], (0,2,3,1))
        print(' * {:15s} | {:20s} | {:10s}'.format('Crop', str(stft.get_shape()), str(stft.dtype)))

        x = tf.abs(stft)
        print(' * {:15s} | {:20s} | {:10s}'.format('Magnitude', str(x.get_shape()), str(x.dtype)))

        downsampling_l = [x]
        for l, nf, fs, st in zip(range(len(n_filters)), n_filters, filter_size, stride):
            name = 'conv{}'.format(l+1)
            x = tfw.conv_2d(x, nf, fs, padding='VALID', activation_fn=tf.nn.relu, stride=st, name=name)
            downsampling_l.append(x)
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        return downsampling_l

    def visual_encoding_ops(self, inp, is_training=True, finetune=False, scope=None):
        vid_units = 512
        inp_shape = tuple(inp.get_shape().as_list())
        print(' * {:15s} | {:20s} | {:10s}'.format('Input', str(inp.get_shape()), str(inp.dtype)))
        x = tf.reshape(inp, shape=(inp_shape[0]*inp_shape[1],) + inp_shape[2:])
        print(' * {:15s} | {:20s} | {:10s}'.format('Reshape', str(x.get_shape()), str(x.dtype)))

        cnn = ResNet18()
        x, ends = cnn.inference_ops(x, finetune, truncate_at='conv5_2')
        init_ops = cnn.restore_pretrained(inp_shape[-1], scope)
        self.init_ops.extend(init_ops)
        self.ends.update([(scope+'/'+key, val) for key, val in ends.iteritems()])
        return x

    def bottleneck_ops(self, x_enc, use_audio=True):
        if len(x_enc) == 0:
            return None
        bottleneck = []
        audio_sz = x_enc[AUDIO][-1].get_shape().as_list()
        for k in [AUDIO, VIDEO, FLOW]:
            if k == AUDIO and not use_audio:
                continue
            if k in x_enc:
                x = x_enc[k][-1] if k == AUDIO else x_enc[k]
                print(' * {:15s} | {:20s} | {:10s}'.format(k+'-feats', str(x.get_shape()), str(x.dtype)))

                if k != AUDIO:
                    name = k+'-fc-red'
                    x = tfw.fully_connected(x, 128, activation_fn=tf.nn.relu, name=name)
                    print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

                sz = x.get_shape().as_list()
                out_shape = (sz[0], sz[1], sz[2]*sz[3]) if k == AUDIO else (sz[0], 1, sz[1]*sz[2]*sz[3])
                x = tf.reshape(x, out_shape)
                print(' * {:15s} | {:20s} | {:10s}'.format(k+'-reshape', str(x.get_shape()), str(x.dtype)))

                name = k+'-fc'
                n_units = 1024 if k == AUDIO else 512
                x = tfw.fully_connected(x, n_units, activation_fn=tf.nn.relu, name=name)
                print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

                if k in [VIDEO, FLOW]:
                    x = tf.tile(x, (1, audio_sz[1], 1))
                    print(' * {:15s} | {:20s} | {:10s}'.format(k+' tile', str(x.get_shape()), str(x.dtype)))

                bottleneck.append(x)

        bottleneck = tf.concat(bottleneck, 2)
        print(' * {:15s} | {:20s} | {:10s}'.format('Concat', str(bottleneck.get_shape()), str(bottleneck.dtype)))
        
        return bottleneck

    def localization_ops(self, x):
        num_out = (self.ambi_order + 1) ** 2 - self.ambi_order ** 2
        num_in = self.ambi_order ** 2

        # Localization
        for i, u in enumerate(self.params.loc_fc_units):
            name = 'fc{}'.format(i+1)
            x = tfw.fully_connected(x, u, activation_fn=tf.nn.relu, name=name)
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
    
        # Compute localization weights
        name = 'fc{}'.format(len(self.params.loc_fc_units)+1)
        x = tfw.fully_connected(
            x, num_out*num_in*(self.params.sep_num_tracks+1), activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
            weight_decay=0, name=name)  # BS x NF x NIN x NOUT

        sz = x.get_shape().as_list()
        x = tf.reshape(x, (sz[0], sz[1], num_out, num_in, self.params.sep_num_tracks+1))
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        sz = x.get_shape().as_list()
        x = tf.tile(tf.expand_dims(x, 2), (1, 1, self.snd_dur/sz[1], 1, 1, 1))
        x = tf.reshape(x, (sz[0], self.snd_dur, sz[2], sz[3], sz[4]))
        print(' * {:15s} | {:20s} | {:10s}'.format('Tile', str(x.get_shape()), str(x.dtype)))

        weights = x[:, :, :, :, :-1]
        print(' * {:15s} | {:20s} | {:10s}'.format('weights', str(weights.get_shape()), str(weights.dtype)))
        biases = x[:, :, :, :, -1]
        print(' * {:15s} | {:20s} | {:10s}'.format('biases', str(biases.get_shape()), str(biases.dtype)))
        return weights, biases

    def separation_ops(self, mono, stft, audio_enc, feats, scope='separation'):
        if self.separation == NO_SEPARATION:
            ss = self.snd_contx / 2
            x_sep = mono[:, :, ss:ss + self.snd_dur]    # BS x 1 x NF
            x_sep = tf.expand_dims(x_sep, axis=1)
            self.ends[scope + '/' + 'all_channels'] = x_sep
            print(' * {:15s} | {:20s} | {:10s}'.format('Crop Audio', str(x_sep.get_shape()), str(x_sep.dtype)))
            return x_sep

        elif self.separation == FREQ_MASK:
            n_filters = [32, 64, 128, 256, 512]
            filter_size = [(7, 16), (3, 7), (3, 5), (3, 5), (3, 5)]
            stride = [(4, 8), (2, 4), (2, 2), (1, 1), (1, 1)]

            name = 'fc-feats'
            feats = tfw.fully_connected(feats, n_filters[-1], activation_fn=tf.nn.relu, name=name)
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(feats.get_shape()), str(feats.dtype)))

            sz = feats.get_shape().as_list()
            enc_sz = audio_enc[-1].get_shape().as_list()
            feats = tf.tile(tf.expand_dims(feats, 2), (1, 1, enc_sz[2], 1))
            feats = tf.reshape(feats, (sz[0], sz[1], enc_sz[2], sz[2]))
            print(' * {:15s} | {:20s} | {:10s}'.format('Tile', str(feats.get_shape()), str(feats.dtype)))

            x = tf.concat([audio_enc[-1], feats], axis=3)
            print(' * {:15s} | {:20s} | {:10s}'.format('Concat', str(x.get_shape()), str(x.dtype)))

            # Up-convolution
            n_chann_in = mono.get_shape().as_list()[1]
            for l, nf, fs, st, l_in in reversed(zip(range(len(n_filters)), [self.params.sep_num_tracks*n_chann_in,]+n_filters[:-1], filter_size, stride, audio_enc[:-1])):
                name = 'deconv{}'.format(l+1)
                x = tfw.deconv_2d(x, nf, fs, stride=st, padding='VALID', activation_fn=None, name=name)
                print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

                if l == 0:
                    break

                x = tf.concat((tf.nn.relu(x), l_in), 3)
                print(' * {:15s} | {:20s} | {:10s}'.format('Concat', str(x.get_shape()), str(x.dtype)))

            # Crop 
            ss = np.floor((self.snd_contx / 2. - self.wind_size) * (4. / self.wind_size))
            tt = np.ceil((self.snd_contx / 2. + self.snd_dur + self.wind_size) * (4. / self.wind_size))
            inp_dim = 95.    # Encoder Dim=1
            skip = (self.snd_contx / 2.) * (4. / self.wind_size)
            skip = int(skip - (inp_dim - 1) / 2.)

            stft = stft[:, :, int(ss):int(tt)]
            print(' * {:15s} | {:20s} | {:10s}'.format('Crop STFT', str(stft.get_shape()), str(stft.dtype)))

            x = x[:, int(ss-skip):int(tt-skip), :]
            print(' * {:15s} | {:20s} | {:10s}'.format('Crop deconv1', str(x.get_shape()), str(x.dtype)))

            x = tf.transpose(x, (0, 3, 1, 2))
            print(' * {:15s} | {:20s} | {:10s}'.format('Permute', str(x.get_shape()), str(x.dtype)))

            x_sz = x.get_shape().as_list()
            x = tf.reshape(x, (x_sz[0], n_chann_in, -1, x_sz[2], x_sz[3]))
            print(' * {:15s} | {:20s} | {:10s}'.format('Reshape', str(x.get_shape()), str(x.dtype)))

            # Apply Mask
            f_mask = tf.cast(tf.sigmoid(x), dtype=tf.complex64)
            print(' * {:15s} | {:20s} | {:10s}'.format('Sigmoid', str(f_mask.get_shape()), str(f_mask.dtype)))

            stft_sep = tf.expand_dims(stft, 2) * f_mask
            print(' * {:15s} | {:20s} | {:10s}'.format('Prod', str(stft_sep.get_shape()), str(stft_sep.dtype)))

            # IFFT
            x_sep = myutils.istft(stft_sep, 4)
            print(' * {:15s} | {:20s} | {:10s}'.format('ISTFT', str(x_sep.get_shape()), str(x_sep.dtype)))

            ss = self.snd_contx / 2.
            skip = np.floor((self.snd_contx / 2. - self.wind_size) * (4. / self.wind_size)) * (self.wind_size / 4.)
            skip += 3. * self.wind_size / 4.    # ISTFT ignores 3/4 of a window
            x_sep = x_sep[:, :, :, int(ss-skip):int(ss-skip)+self.snd_dur]
            print(' * {:15s} | {:20s} | {:10s}'.format('Crop', str(x_sep.get_shape()), str(x_sep.dtype)))

        else:
            raise ValueError('Unknown separation mode.')

        self.ends[scope + '/' + 'all_channels'] = x_sep
        return x_sep

    def inference_ops(self, audio, video=None, flow=None, is_training=True):
        audio = tf.transpose(audio, (0, 2, 1))  # BATCH_SIZE x N_CHANNELS x N_FRAMES

        tensors = [audio, video, flow]
        names = ['audio', 'video', 'flow']
        print('Inputs')
        for t, n in zip(tensors, names):
            if t is not None:
                self.ends[n] = t
                print(' * {:15s} | {:20s} | {:10s}'.format(n, str(t.get_shape()), str(t.dtype)))

        # STFT (0.025s windows, 25% hop)
        print('\nSTFT')
        stft = myutils.stft(audio, self.wind_size, 4)
        print(' * {:15s} | {:20s} | {:10s}'.format('Mono', str(audio.get_shape()), str(audio.dtype)))
        print(' * {:15s} | {:20s} | {:10s}'.format('STFT', str(stft.get_shape()), str(stft.dtype)))

        x_enc = {}

        # Audio encoder
        if AUDIO in self.encoders:
            print('\nAudio encoder')
            scope = 'audio_encoder'
            with tf.variable_scope(scope):
                x_enc[AUDIO] = self.audio_encoder_ops(stft)
        
        # Video encoder
        if VIDEO in self.encoders:
            print('\nVideo encoder')
            scope = 'video_encoder'
            with tf.variable_scope(scope):
                x_enc[VIDEO] = self.visual_encoding_ops(
                    video, is_training=is_training, finetune=True, scope=scope)

        # Flow encoder
        if FLOW in self.encoders:
            print('\nFlow encoder')
            scope = 'flow_encoder'
            with tf.variable_scope(scope):
                x_enc[FLOW] = self.visual_encoding_ops(
                    flow, is_training=is_training, finetune=True, scope=scope)

        # Mixer
        print('\nBottleneck')
        scope = 'bottleneck'
        with tf.variable_scope(scope):
            feats = self.bottleneck_ops(x_enc, AUDIO in self.encoders)

        # Localization coefficients
        scope = 'localization'
        print('\n Localization')
        with tf.variable_scope(scope):
            weights, biases = self.localization_ops(feats)
            
        self.loc_channels = [weights, biases]

        # Source separation
        scope = 'separation'
        print('\n Separation')
        with tf.variable_scope(scope):
            x_sep = self.separation_ops(audio, stft, x_enc[AUDIO] if len(x_enc) else None, feats, scope)
            
        self.sep_channels = x_sep
        self.inp_spect = tf.abs(stft)
            
        # Decode ambisonics
        scope = 'decoder'
        print('\n Ambix Generation')
        x_sep = tf.transpose(x_sep, (0, 3, 1, 2))
        print(' * {:15s} | {:20s} | {:10s}'.format('Input Audio', str(x_sep.get_shape()), str(x_sep.dtype)))
        print(' * {:15s} | {:20s} | {:10s}'.format('Input Weights', str(weights.get_shape()), str(weights.dtype)))
        print(' * {:15s} | {:20s} | {:10s}'.format('Input Biases', str(biases.get_shape()), str(biases.dtype)))
        with tf.variable_scope(scope):
            # Predict ambisonics (A_t = W_t*s_t + b_t)
            x_ambi = tf.reduce_sum(tf.reduce_sum(weights * tf.expand_dims(x_sep, axis=2), axis=4), axis=3) + biases[:,:,:,0]
            self.ends[scope + '/ambix'] = x_ambi
            print(' * {:15s} | {:20s} | {:10s}'.format('Ambix', str(x_ambi.get_shape()), str(x_ambi.dtype)))

        return x_ambi
