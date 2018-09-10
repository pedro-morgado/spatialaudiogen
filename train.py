"""Description
"""
import argparse
import os
import sys
import math
import time
import tensorflow as tf
from feeder import Feeder
from model import SptAudioGen, SptAudioGenParams
import myutils
from collections import deque
from definitions import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('db_dir', help='Directory containing db.')
    parser.add_argument('model_dir', help='Directory to store model.')
    parser.add_argument('--subset_fn',   default='')

    parser.add_argument('--encoders', nargs='*', type=str.lower, choices=ENCODERS, default=['audio', 'flow', 'video'], help="List of encoders.")
    parser.add_argument('--separation', type=str.lower, default=FREQ_MASK, choices=SEPARATION, help="Separation net architecture.")

    parser.add_argument('--ambi_order',   type=int,   default=1, help="Ambisonics order")
    parser.add_argument('--audio_rate',   type=int,   default=48000, help="Audio frame rate")
    parser.add_argument('--video_rate',   type=int,   default=10, help="Video frame rate")
    parser.add_argument('--context',      type=float, default=1.0, help="Context duration")
    parser.add_argument('--sample_dur',   type=float, default=0.1, help="Training sample duration")

    parser.add_argument('--n_iters',      type=int,   default=1000000, help="Number of iterations")
    parser.add_argument('--lr',           type=float, default=1e-4, help="Base learning rate")
    parser.add_argument('--lr_decay',     type=float, default=0.5, help="Learning rate decay")
    parser.add_argument('--lr_iters',     type=int, default=250000, help="Iterations between decays.")
    parser.add_argument('--batch_size',   type=int,   default=32, help="Batch size")
    parser.add_argument('--resume', action='store_true', help="Restore and resume training.")

    parser.add_argument('--num_sep_tracks', default=NUM_SEP_TRACKS_DEF, type=int,
                        help="Number of separataion tracks.")
    parser.add_argument('--fft_window', default=SEP_FFT_WINDOW_DEF, type=float,
                        help="Window size for fft computation (secs).")
    parser.add_argument('--context_units', default=CTX_FEATS_FCUNITS_DEF, nargs='+', type=int,
                        help="Number of fully connected units for context feature generation.")
    parser.add_argument('--freq_mask_units', default=SEP_FREQ_MASK_FCUNITS_DEF, nargs='*', type=int,
                        help="Number of fully connected units for frequency mask generation.")
    parser.add_argument('--loc_units', default=LOC_FCUNITS_DEF, nargs='+', type=int,
                        help="Number of fully connected units for localization weights generation.")

    parser.add_argument('--gpu',          type=int,   default=0, help="GPU id")

    args = parser.parse_args(sys.argv[1:])
    if len(args.subset_fn) == 0:
        args.subset_fn = None
    if args.resume and not os.path.isfile(os.path.join(args.model_dir, 'train-params.txt')):
        args.resume = False
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    print('\n' + '='*30 + ' ARGUMENTS ' + '='*30)
    sys.stdout.flush()
    if args.resume:
        params = myutils.load_params(args.model_dir)
        args.encoders = params.encoders
        args.separation = params.separation
        args.ambi_order = params.ambi_order
        args.audio_rate = params.audio_rate
        args.video_rate = params.video_rate
        args.context = params.context
        args.sample_dur = params.sample_dur
    else:
        myutils.save_params(args)
    myutils.print_params(args)

    # Feeder
    min_t = min([args.context, args.sample_dur, 1./args.video_rate])
    args.video_rate = int(1. / min_t)
    with tf.device('/cpu:0'), tf.variable_scope('feeder'):
        feeder = Feeder(args.db_dir,
                        subset_fn=args.subset_fn,
                        ambi_order=args.ambi_order,
                        audio_rate=args.audio_rate,
                        video_rate=args.video_rate,
                        context=args.context,
                        duration=args.sample_dur,
                        return_video=VIDEO in args.encoders,
                        img_prep=myutils.img_prep_fcn(),
                        return_flow=FLOW in args.encoders,
                        frame_size=(224, 448),
                        queue_size=args.batch_size*5,
                        n_threads=4,
                        for_eval=False)

        batches = feeder.dequeue(args.batch_size)
        ambix_batch = batches['ambix']
        video_batch = batches['video'] if 'video' in args.encoders else None
        flow_batch = batches['flow'] if 'flow' in args.encoders else None
        audio_mask_batch = batches['audio_mask']

        t = int(args.audio_rate * args.sample_dur)
        ss = int(args.audio_rate * args.context) / 2
        n_chann_in = args.ambi_order**2
        audio_input = ambix_batch[:, :, :n_chann_in]
        audio_target = ambix_batch[:, ss:ss+t, n_chann_in:]

    print('\n' + '=' * 20 + ' MODEL ' + '=' * 20)
    sys.stdout.flush()
    with tf.device('/gpu:0'):
        # Model
        num_sep = args.num_sep_tracks if args.separation != NO_SEPARATION else 1
        params = SptAudioGenParams(sep_num_tracks=num_sep, ctx_feats_fc_units=args.context_units,
                                   loc_fc_units=args.loc_units, sep_freq_mask_fc_units=args.freq_mask_units,
                                   sep_fft_window=args.fft_window)
        model = SptAudioGen(ambi_order=args.ambi_order, 
                            audio_rate=args.audio_rate,
                            video_rate=args.video_rate, 
                            context=args.context,
                            sample_duration=args.sample_dur, 
                            encoders=args.encoders,
                            separation=args.separation,
                            params=params)
        ambix_pred = model.inference_ops(audio=audio_input, video=video_batch, flow=flow_batch, is_training=True)
        
        # Losses and evaluation metrics
        print(audio_mask_batch)
        with tf.variable_scope('metrics'):
            metrics_t, _, _, _, _ = model.evaluation_ops(ambix_pred, audio_target, audio_input[:, ss:ss+t],
                                             mask_channels=audio_mask_batch[:, args.ambi_order**2:])

        step_t = tf.Variable(0, trainable=False, name='step')
        with tf.variable_scope('loss'):
            loss_t = model.loss_ops(metrics_t, step_t)
            losses_t = {l: loss_t[l] for l in loss_t}
            regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if regularizers and 'regularization' in losses_t:
                losses_t['regularization'] = tf.add_n(regularizers)
            losses_t['total_loss'] = tf.add_n(losses_t.values())

        # Optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('optimization') and tf.control_dependencies(update_ops):
            train_op, lr_t = myutils.optimize(losses_t['total_loss'], step_t, args)

        # Initialization
        rest_ops = model.init_ops
        init_op = [tf.global_variables_initializer(),
                   tf.local_variables_initializer()]
        saver = tf.train.Saver(max_to_keep=1)

        # Tensorboard
        metrics_t['training_loss'] = losses_t['total_loss']
        metrics_t['queue'] = feeder.queue_state
        metrics_t['lr'] = lr_t
        myutils.add_scalar_summaries(metrics_t.values(), metrics_t.keys())
        summary_ops = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_writer = tf.summary.FileWriter(args.model_dir, flush_secs=30)
        #summary_writer.add_graph(tf.get_default_graph())

    print('\n' + '='*30 + ' VARIABLES ' + '='*30)
    model_vars = tf.global_variables()
    import numpy as np
    for v in model_vars:
        if 'Adam' in v.op.name.split('/')[-1]:
            continue
        print(' * {:50s} | {:20s} | {:7s} | {:10s}'.format(v.op.name, str(v.get_shape()), str(np.prod(v.get_shape())), str(v.dtype)))

    print('\n' + '='*30 + ' TRAINING ' + '='*30)
    sys.stdout.flush()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    with tf.Session(config=config) as sess:
        print('Initializing network...')
        sess.run(init_op)
        if rest_ops:
            sess.run(rest_ops)

        print('Initializing data feeders...')
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        feeder.start_threads(sess)

        tf.get_default_graph().finalize()

        # Restore model
        init_step = 0
        if args.resume:
            print('Restoring previously saved model...')
            ckpt = tf.train.latest_checkpoint(args.model_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                init_step = sess.run(step_t)

        try:
            print('Start training...')
            duration = deque(maxlen=20)
            for step in range(init_step, args.n_iters):
                start_time = time.time()
                if step % 20 != 0:
                    sess.run(train_op)
                else:
                    outs = sess.run([train_op, summary_ops, losses_t['total_loss']] +
                                     losses_t.values() + metrics_t.values())
                    if math.isnan(outs[2]):
                        raise ValueError('Training produced a NaN metric or loss.')
                duration.append(time.time() - start_time)


                if step % 20 == 0:     # Print progress to terminal and tensorboard
                    myutils.print_stats(outs[3:], losses_t.keys() + metrics_t.keys(),
                                        args.batch_size, duration, step, tag='TRAIN')
                    summary_writer.add_summary(outs[1], step)
                    sys.stdout.flush()

                if step % 5000 == 0 and step != 0:  # Save checkpoint
                    saver.save(sess, args.model_dir+'/model.ckpt', global_step=step_t)
                    print('='*60 + '\nCheckpoint saved\n' + '='*60)

        except Exception, e:
            print(str(e))

        finally:
            print('End of training.')
            print('Saving model.')
            myutils.save_params(args)
            saver.save(sess, args.model_dir+'/model.ckpt')
            coord.request_stop()
            coord.join(stop_grace_period_secs=10)


if __name__ == '__main__':
    main(parse_arguments())
