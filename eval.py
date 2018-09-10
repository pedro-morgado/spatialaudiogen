"""Description
"""
import sys, os, time, argparse
from collections import OrderedDict, deque
import tensorflow as tf
import numpy as np

from feeder import Feeder
from model import SptAudioGen, SptAudioGenParams
from pyutils.ambisonics.distance import ambix_emd
import myutils
from definitions import *

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_dir', help='Directory to store model.')
    parser.add_argument('--subset_fn', default='')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help="GPU id")
    args = parser.parse_args(sys.argv[1:])
    if len(args.subset_fn) == 0:
        args.subset_fn = None
    return args


def main(args):
    eval_fn = os.path.join(args.model_dir, 'eval-detailed.txt')
    assert os.path.exists(args.model_dir), 'Model dir does not exist.'
    assert args.overwrite or not os.path.exists(eval_fn), 'Evaluation file already exists.'
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    print '\n' + '='*30 + ' ARGUMENTS ' + '='*30
    params = myutils.load_params(args.model_dir)
    for k, v in params.__dict__.iteritems():
        print 'TRAIN | {}: {}'.format(k, v)
    for k, v in args.__dict__.iteritems():
        print 'EVAL | {}: {}'.format(k, v)
    sys.stdout.flush()

    DURATION = 0.1
    BATCH_SIZE = 16
    with tf.device('/cpu:0'), tf.variable_scope('feeder'):
        feeder = Feeder(params.db_dir,
                        subset_fn=args.subset_fn,
                        ambi_order=params.ambi_order,
                        audio_rate=params.audio_rate,
                        video_rate=params.video_rate,
                        context=params.context,
                        duration=DURATION,
                        return_video=VIDEO in params.encoders,
                        img_prep=myutils.img_prep_fcn(),
                        return_flow=FLOW in params.encoders,
                        frame_size=(224, 448),
                        queue_size=BATCH_SIZE*5,
                        n_threads=4,
                        for_eval=True)
        batches = feeder.dequeue(BATCH_SIZE)

        ambix_batch = batches['ambix']
        video_batch = batches['video'] if VIDEO in params.encoders else None
        flow_batch = batches['flow'] if FLOW in params.encoders else None
        audio_mask_batch = batches['audio_mask']

        ss = int(params.audio_rate * params.context) / 2
        t = int(params.audio_rate * DURATION)
        audio_input = ambix_batch[:, :, :params.ambi_order**2]
        audio_target = ambix_batch[:, ss:ss+t, params.ambi_order**2:]

    print '\n' + '=' * 20 + ' MODEL ' + '=' * 20
    sys.stdout.flush()
    with tf.device('/gpu:0'):
        # Model
        num_sep = params.num_sep_tracks if params.separation != NO_SEPARATION else 1
        net_params = SptAudioGenParams(sep_num_tracks=num_sep, ctx_feats_fc_units=params.context_units,
                                       loc_fc_units=params.loc_units, sep_freq_mask_fc_units=params.freq_mask_units,
                                       sep_fft_window=params.fft_window)
        model = SptAudioGen(ambi_order=params.ambi_order, 
                            audio_rate=params.audio_rate,
                            video_rate=params.video_rate, 
                            context=params.context,
                            sample_duration=DURATION, 
                            encoders=params.encoders,
                            separation=params.separation,
                            params=net_params)

        # Inference
        pred_t = model.inference_ops(audio=audio_input, video=video_batch, flow=flow_batch, is_training=False)

        # Losses and evaluation metrics
        with tf.variable_scope('metrics'):
            w_t = audio_input[:, ss:ss+t]
            _, stft_dist_ps, lsd_ps, mse_ps, snr_ps = model.evaluation_ops(pred_t, audio_target, w_t,
                                             mask_channels=audio_mask_batch[:, params.ambi_order**2:])
        # Loader
        vars2save = [v for v in tf.global_variables() if not v.op.name.startswith('metrics')]
        saver = tf.train.Saver(vars2save)

    print '\n' + '='*30 + ' VARIABLES ' + '='*30
    model_vars = tf.global_variables()
    import numpy as np
    for v in model_vars:
        if 'Adam' in v.op.name.split('/')[-1]:
            continue
        print ' * {:50s} | {:20s} | {:7s} | {:10s}'.format(v.op.name, str(v.get_shape()), str(np.prod(v.get_shape())), str(v.dtype))

    print '\n' + '='*30 + ' EVALUATION ' + '='*30
    sys.stdout.flush()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    with tf.Session(config=config) as sess:
        print 'Loading model...'
        sess.run(model.init_ops)
        saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))

        print 'Initializing data feeders...'
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        feeder.start_threads(sess)

        all_metrics = ['amplitude/predicted', 'amplitude/gt', 
            'mse/avg', 'mse/X','mse/Y', 'mse/Z', 
            'stft/avg', 'stft/X','stft/Y', 'stft/Z', 
            'lsd/avg', 'lsd/X','lsd/Y', 'lsd/Z', 
            'mel_lsd/avg', 'mel_lsd/X','mel_lsd/Y', 'mel_lsd/Z', 
            'snr/avg', 'snr/X','snr/Y', 'snr/Z', 
            'env_mse/avg', 'env_mse/X','env_mse/Y', 'env_mse/Z', 
            'emd/dir', 'emd/dir2']
        metrics = OrderedDict([(key, []) for key in all_metrics])
        sample_ids = []
        telapsed = deque(maxlen=20)

        print 'Start evaluation...'
        it = -1
        # run_options = tf.RunOptions(timeout_in_ms=60*1000)
        while True:
            it += 1
            if feeder.done(sess):
                break
            start_time = time.time()
            outs = sess.run([batches['id'], audio_mask_batch, w_t, audio_target, pred_t, stft_dist_ps, lsd_ps, mse_ps, snr_ps])
            video_id, layout, mono, gt, pred = outs[:5]
            gt_m = np.concatenate((mono, gt), axis=2) * layout[:, np.newaxis, :]
            pred_m = np.concatenate((mono, pred), axis=2) * layout[:, np.newaxis, :]
            stft_dist, lsd, mse, snr = outs[5:]

            _env_time = 0.
            _emd_time = 0.
            _pow_time = 0.
            _lsd_time = 0.
            for smp in range(BATCH_SIZE):
                metrics['stft/avg'].append(np.mean(stft_dist[smp]))
                for i, ch in zip(range(3), 'YZX'):
                    metrics['stft/'+ch].append(stft_dist[smp, i])
 
                metrics['lsd/avg'].append(np.mean(lsd[smp]))
                for i, ch in zip(range(3), 'YZX'):
                    metrics['lsd/'+ch].append(lsd[smp, i])

                metrics['mse/avg'].append(np.mean(mse[smp]))
                for i, ch in zip(range(3), 'YZX'):
                    metrics['mse/'+ch].append(mse[smp, i])

                metrics['snr/avg'].append(np.nanmean(snr[smp]))
                for i, ch in zip(range(3), 'YZX'):
                    metrics['snr/'+ch].append(snr[smp, i])

                # Compute Mel LSD distance
                _t = time.time()
                mel_lsd = myutils.compute_lsd_dist(pred[smp], gt[smp], params.audio_rate) 
                metrics['mel_lsd/avg'].append(np.mean(mel_lsd))
                for i, ch in zip(range(3), 'YZX'):
                    metrics['mel_lsd/'+ch].append(mel_lsd[i])
                _lsd_time += (time.time() - _t)

                # Compute envelope distances
                _t = time.time()
                env_mse = myutils.compute_envelope_dist(pred[smp], gt[smp]) 
                metrics['env_mse/avg'].append(np.mean(env_mse))
                for i, ch in zip(range(3), 'YZX'):
                    metrics['env_mse/'+ch].append(env_mse[i])
                _env_time += (time.time() - _t)

                # Compute EMD (for speed, only compute emd over first 0.1s of every 1sec)
                _t = time.time()
                emd_dir, emd_dir2 = ambix_emd(pred_m[smp], gt_m[smp], model.snd_rate, ang_res=30)
                metrics['emd/dir'].append(emd_dir)
                metrics['emd/dir2'].append(emd_dir2)
                _emd_time += (time.time() - _t)

                # Compute chunk power
                _t = time.time()
                metrics['amplitude/gt'].append(np.abs(gt[smp]).max())
                metrics['amplitude/predicted'].append(np.abs(pred[smp]).max())
                _pow_time += (time.time() - _t)

                sample_ids.append(video_id[smp])

            telapsed.append(time.time() - start_time)
            #print '\nTotal:', telapsed[-1]
            #print 'Env:', _env_time
            #print 'LSD:', _lsd_time
            #print 'EMD:', _emd_time
            #print 'POW:', _pow_time

            if it % 100 == 0:
                # Store evaluation metrics
                with open(eval_fn, 'w') as f:
                    f.write('SampleID | {}\n'.format(' '.join(metrics.keys())))
                    for smp in range(len(sample_ids)):
                        f.write('{} | {}\n'.format(sample_ids[smp], ' '.join([str(metrics[key][smp]) for key in metrics])))

            if it % 5 == 0:
                stats = OrderedDict([(m, np.mean(metrics[m])) for m in all_metrics])
                myutils.print_stats(stats.values(), stats.keys(), BATCH_SIZE, telapsed, it, tag='EVAL')
                sys.stdout.flush()

        # Print progress
        stats = OrderedDict([(m, np.mean(metrics[m])) for m in all_metrics])
        myutils.print_stats(stats.values(), stats.keys(), BATCH_SIZE, telapsed, it, tag='EVAL')
        sys.stdout.flush()
        with open(eval_fn, 'w') as f:
            f.write('SampleID | {}\n'.format(' '.join(metrics.keys())))
            for smp in range(len(sample_ids)):
                f.write('{} | {}\n'.format(sample_ids[smp], ' '.join([str(metrics[key][smp]) for key in metrics])))

        print('\n'+'#'*60)
        print('End of evaluation.')


if __name__ == '__main__':
    main(parse_arguments())
