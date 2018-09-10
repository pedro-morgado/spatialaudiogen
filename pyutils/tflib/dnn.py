"""
DNN framework
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import tensorflow as tf
import numpy as np
from datetime import datetime
import string, random
from pyutils.tflib.inputs import PythonFeeder
TRAIN = 0
TEST = 1

DTYPE_DICT = {'int32': tf.int32, 'int64': tf.int64, 'float32': tf.float32, 'float64': tf.float32}

class DNN(object):
    def __init__(self, tb_dir="/tmp/tflearn_logs/", tb_verbosity=0, ckp_path=None):
        self.sess_cfg = {'gpu_fraction': 1.0, 'growth': True, 'soft_placement': True, 'log_device': False, 'num_workers': 2}

        # variables used in training mode
        self.graph = tf.Graph()
        self.sess = None
        self.global_step = None
        self.epoch = None

        self.batch_size = None
        self.batches_per_epoch = None
        self.train_feeders = None
        self.val_feeders = None
        self.trainer = None
        self.validator = None
        self.saver = None

        self.tb_dir = tb_dir
        self.tb_verbosity = tb_verbosity
        self.ckp_path = ckp_path

        # variables used in deploy mode
        self.train_inputs_t = None
        self.train_targets_t = None
        self.val_inputs_t = None
        self.val_targets_t = None
        self.logits_t = None
        self.restorer = None
        self.input_info = None

    def __del__(self):
        self._close_session()

    def _start_session(self):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.sess_cfg['gpu_fraction'],
            allow_growth=self.sess_cfg['growth'],
            allocator_type="BFC"
        )
        optimizer_opts = tf.OptimizerOptions(do_common_subexpression_elimination=True,
                                             do_constant_folding=True,
                                             do_function_inlining=True)
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=self.sess_cfg['log_device'],
            intra_op_parallelism_threads=self.sess_cfg['num_workers'],
            inter_op_parallelism_threads=self.sess_cfg['num_workers'],
            graph_options=tf.GraphOptions(optimizer_options=optimizer_opts),
            gpu_options=gpu_options)
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session(graph=self.graph, config=config)

    def _close_session(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def _start_new_graph(self):
        self.graph = tf.Graph()

    def _save_checkpoint(self, snapshot_step=None, snapshot_epoch=None, force=False, with_step=False):
        if self.ckp_path is None:
           return
        curr_step = self.global_step.eval(session=self.sess)
        curr_epoch = self.epoch.eval(session=self.sess)
        if force or (snapshot_epoch > 0 and curr_epoch % snapshot_epoch == 0):
            self.saver.save(self.sess, self.ckp_path, global_step=self.global_step if with_step else None)
        elif force or (snapshot_step > 0 and curr_step % snapshot_step == 0):
            self.saver.save(self.sess, self.ckp_path, global_step=self.global_step if with_step else None)

    def validate(self, validate_step=None, force=False):
        if self.validator is None:
            return
        curr_step = self.global_step.eval(session=self.sess)
        if force or (validate_step > 0 and (curr_step % validate_step == 0 or curr_step == 1)):
            if curr_step in self.val_steps:
                return
            self.val_steps.append(curr_step)
            self.validator.step(self.sess)

    def configure_session(self,
                          gpu_fraction=1.0,
                          growth=True,
                          log_device=True,
                          num_workers=2):
        self.sess_cfg['gpu_fraction'] = gpu_fraction
        self.sess_cfg['growth'] = growth
        self.sess_cfg['log_device'] = log_device
        self.sess_cfg['num_workers'] = num_workers

    def training_setup(self, train_feeders, model,
                       optimizer=tf.train.AdamOptimizer(0.001),
                       clip_grads=-1,
                       val_feeders=None,
                       val_steps=20,
                       snapshot_dir=None,
                       resume=False,
                       gpu_dev=0):
        gpu = '/gpu:%d' % gpu_dev

        self.model = model
        if train_feeders is not None:
            self.train_feeders = train_feeders
        if val_feeders is not None:
            self.val_feeders = val_feeders

        self._close_session()
        self._start_new_graph()
        with self.graph.as_default():
            # Global variables
            with tf.device('/cpu:0'):
                with tf.variable_scope('step'):
                    self.global_step = tf.get_variable('global_step', (), initializer=tf.constant_initializer(0), trainable=False)

                with tf.variable_scope('epoch'):
                    self.epoch = tf.get_variable('epoch', (), initializer=tf.constant_initializer(0), trainable=False)
                    self.incr_epoch = tf.assign(self.epoch, tf.add(self.epoch, 1))

                reset_step = tf.assign(self.global_step, 0)
                reset_epoch = tf.assign(self.epoch, 0)

            # Training inputs
            if train_feeders is not None:
                with tf.variable_scope('train_input'):
                    if isinstance(train_feeders, dict):
                        train_inputs_t, train_targets_t = {}, {}
                        for key, feeder in train_feeders.iteritems():
                            train_inputs_t[key], train_targets_t[key] = feeder.build_batch(gpu)
                        train_batch_size, num_samples = train_feeders.values()[0].batch_size, train_feeders.values()[0].num_samples
                    else:
                        train_inputs_t, train_targets_t = train_feeders.build_batch(gpu)
                        train_batch_size, num_samples = train_feeders.batch_size, train_feeders.num_samples
                    self.batches_per_epoch = int(num_samples / train_batch_size)

            # Validation inputs
            if val_feeders is not None:
                with tf.variable_scope('valid_input'):
                    if isinstance(val_feeders, dict):
                        val_inputs_t, val_targets_t = {}, {}
                        for key, feeder in val_feeders.iteritems():
                            val_inputs_t[key], val_targets_t[key] = feeder.build_batch(gpu)
                        val_batch_size, num_samples = val_feeders.values()[0].batch_size, val_feeders.values()[0].num_samples
                    else:
                        val_inputs_t, val_targets_t = val_feeders.build_batch(gpu)
                        val_batch_size = val_feeders.batch_size

            with tf.device(gpu):
                with tf.variable_scope('inference'):
                    if train_feeders is not None:
                        train_logits_t = model.inference_ops(train_inputs_t, is_training=True, reuse=False)
                        self.train_logits_t = train_logits_t
                    if val_feeders is not None:
                        val_logits_t = model.inference_ops(val_inputs_t, is_training=False, reuse=train_feeders is not None)

                if train_feeders is not None:
                    with tf.variable_scope('loss'):
                        train_loss_t = model.loss_ops(train_logits_t, train_targets_t)

                    with tf.variable_scope('eval'):
                        train_metrics_t = model.evaluation_ops(train_logits_t, train_targets_t)

                    with tf.variable_scope('TrainOp'):
                        self.trainer = TrainOp(train_loss_t, optimizer, clip_gradients=clip_grads, metrics=train_metrics_t,
                                               batch_size=train_batch_size, batch_per_epoch=self.batches_per_epoch,
                                               step_tensor=self.global_step, epoch_tensor=self.epoch,
                                               tb_verbosity=self.tb_verbosity, logdir=os.path.join(self.tb_dir, 'train'))

                if val_feeders is not None:
                    with tf.variable_scope('EvalOp'):
                        with tf.variable_scope('loss'):
                            val_loss_t = model.loss_ops(val_logits_t, val_targets_t)

                        with tf.variable_scope('eval'):
                            val_metrics_t = model.evaluation_ops(val_logits_t, val_targets_t)

                        self.validator = EvaluateOp(val_loss_t, val_metrics_t, num_steps=val_steps, batch_size=val_batch_size,
                                                    step_tensor=self.global_step, epoch_tensor=self.epoch,
                                                    logdir=os.path.join(self.tb_dir, 'valid'))
                init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                initialize_ops = getattr(model, 'initialize_ops', None)
                model_init_op = None
                if callable(initialize_ops):
                    model_init_op = initialize_ops()

            with tf.device('/cpu:0'):
                self.saver = tf.train.Saver()

        # Start session and initialize variables
        self._start_session()
        self.val_steps = []

        if snapshot_dir is not None and os.path.exists(os.path.join(snapshot_dir, 'checkpoint')):
            # print([var.op.name for var in self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
            # exit(0)
            self.sess.run(init_op)
            restorer = tf.train.Saver(self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inference'))
            restorer.restore(self.sess, tf.train.latest_checkpoint(snapshot_dir))
            if not resume:
                self.sess.run([reset_epoch, reset_step])
        else:
            self.sess.run(init_op)
            if model_init_op:
                self.sess.run(model_init_op)

        self.graph.finalize()
        # Memorize input tensors for debugging
        if train_feeders is not None:
            self.train_inputs_t = train_inputs_t
            self.train_targets_t = train_targets_t
        if val_feeders is not None:
            self.val_inputs_t = val_inputs_t
            self.val_targets_t = val_targets_t

    def start_feeders(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        # Training feeders
        if self.train_feeders:
            if isinstance(self.train_feeders, dict):
                [feeder.start_feeder(self.sess) for feeder in self.train_feeders.itervalues()]
            else:
                self.train_feeders.start_feeder(self.sess)

        # Validation feeders
        if self.val_feeders:
            if isinstance(self.val_feeders, dict):
                [feeder.start_feeder(self.sess) for feeder in self.val_feeders.itervalues()]
            else:
                self.val_feeders.start_feeder(self.sess)

    def stop_feeders(self):
        # Training feeders
        if self.train_feeders:
            if isinstance(self.train_feeders, dict):
                [feeder.request_stop() for feeder in self.train_feeders.itervalues()]
                [feeder.join() for feeder in self.train_feeders.itervalues()]
            else:
                self.train_feeders.request_stop()
                self.train_feeders.join()

        # Validation feeders
        if self.val_feeders:
            if isinstance(self.train_feeders, dict):
                [feeder.request_stop() for feeder in self.val_feeders.itervalues()]
                [feeder.join() for feeder in self.val_feeders.itervalues()]
            else:
                self.val_feeders.request_stop()
                self.val_feeders.join()

        # Stop tensorflow queues
        self.coord.request_stop()
        self.coord.join(self.threads)


    def train(self, n_epoch=10,
              snapshot_step=0,
              snapshot_epoch=False,
              validation_step=0,
              validation_epoch=True):

        # Start input enqueue threads.
        self.start_feeders()

        try:
            # Train
            safe_exit = False
            step = self.sess.run(self.global_step)
            epoch = int(step)/int(self.batches_per_epoch)
            while step < n_epoch*int(self.batches_per_epoch):
                if int(step-1) % self.batches_per_epoch == 0:
                    self.sess.run(self.incr_epoch)
                print('='*20+'   Epoch %d   ' % (epoch+1) + '='*20)

                for batch_step in range(int(step) % int(self.batches_per_epoch), self.batches_per_epoch):
                    # if coord.should_stop():
                    #    break
                    self.trainer.step(self.sess)
                    self._save_checkpoint(snapshot_step=snapshot_step, with_step=True)
                    self.validate(validation_step)

                step = self.sess.run(self.global_step)
                epoch = int(step)/int(self.batches_per_epoch)

                # if coord.should_stop():
                #     break
                self._save_checkpoint(snapshot_epoch=snapshot_epoch, with_step=True)
                self.validate(validation_step, validation_epoch)
                print('')
            safe_exit = True

        except KeyboardInterrupt:
            safe_exit = True

        if safe_exit:
            # When done, save checkpoint.
            self._save_checkpoint(force=True, with_step=False)

        self.stop_feeders()
        self._close_session()

    def deploy_setup(self, input_info, model, checkpoint,
                     batch_size=256,
                     gpu_dev=0):
        gpu = '/gpu:%d' % gpu_dev
        self.input_info = input_info
        self.batch_size = batch_size

        self._close_session()
        self._start_new_graph()
        with self.graph.as_default():
            with tf.device(gpu):
                with tf.variable_scope('inputs'):
                    if 'dtype' in input_info.values()[0]:
                        names = input_info.keys()
                        dtypes = [DTYPE_DICT[val['dtype']] for _, val in input_info.iteritems()]
                        shapes = [val['shape'] for _, val in input_info.iteritems()]
                        self.inputs_t = [tf.placeholder(dtype=d, shape=(None,) + s, name=n) for n, d, s in zip(names, dtypes, shapes)]
                    else:
                        self.inputs_t = {}
                        for phase in input_info:
                            names = input_info[phase].keys()
                            dtypes = [DTYPE_DICT[val['dtype']] for _, val in input_info[phase].iteritems()]
                            shapes = [val['shape'] for _, val in input_info[phase].iteritems()]
                            self.inputs_t[phase] = [tf.placeholder(dtype=d, shape=(None,) + s, name=n) for n, d, s in zip(names, dtypes, shapes)]

                with tf.variable_scope('inference'):
                    self.logits_t = model.inference_ops(self.inputs_t, is_training=False)   #tf.constant(False, name='IsTraining'))

            with tf.device('/cpu:0'):
                self.restorer = tf.train.Saver(self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inference'))

        self.graph.finalize()
        self._start_session()
        self.restorer.restore(self.sess, checkpoint)

    def predict(self, inputs, outputs_list=None):
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == len(self.input_info)
            inputs = {key: inputs[i] for i, key in enumerate(self.input_info.keys())}
        if outputs_list is None:
            outputs_list = [self.logits_t]
        if not isinstance(outputs_list, (list, tuple)):
            outputs_list = [outputs_list]

        num_samples = inputs.values()[0].shape[0]
        batch_lims = range(0, num_samples, self.batch_size)
        outputs = [[] for _ in range(len(outputs_list))]
        for lim in batch_lims:
            feed_dict = {self.inputs_t[i]: inputs[key][lim:lim+self.batch_size] for i, key in enumerate(self.input_info.keys())}
            outs = self.sess.run(outputs_list, feed_dict=feed_dict)
            for i, o in enumerate(outs):
                outputs[i].append(o)
        for i in range(len(outputs)):
            outputs[i] = np.concatenate(outputs[i], 0)
        return outputs

    def save_model(self, fn):
        self.saver.save(self.sess, fn)


class TrainOp(object):
    """ TrainOp.

    TrainOp represents a set of operation used for optimizing a network.

    A TrainOp is meant to hold all training parameters of an optimizer.
    `Trainer` class will then instantiate them all specifically considering all
    optimizers of the network (set names, scopes... set optimization ops...).

    Arguments:
        loss: `Tensor`. Loss operation to evaluate network cost.
            Optimizer will use this cost function to train network.
        optimizer: `Optimizer`. Tensorflow Optimizer. The optimizer to
            use to train network.
        metric:  `Tensor`. The metric tensor to be used for evaluation.
        batch_size: `int`. Batch size for datalib feeded to this optimizer.
            Default: 64.
        ema: `float`. Exponential moving averages.
        trainable_vars: list of `tf.Variable`. List of trainable variables to
            use for training. Default: all trainable variables.
        shuffle: `bool`. Shuffle datalib.
        step_tensor: `tf.Tensor`. A variable holding training step. If not
            provided, it will be created. Early defining the step tensor
            might be useful for network creation, such as for learning rate
            decay.
        validation_monitors: `list` of `Tensor` objects.  List of variables
            to compute during validation, which are also used to produce
            summaries for output to TensorBoard.  For example, this can be
            used to periodically record a confusion matrix or AUC metric,
            during training.  Each variable should have rank 1, i.e.
            shape [None].
        validation_batch_size: `int` or None. If `int`, specifies the batch
            size to be used for the validation datalib feed; otherwise
            defaults to being th esame as `batch_size`.
        name: `str`. A name for this class (optional).
        graph: `tf.Graph`. Tensorflow Graph to use for training. Default:
            default tf graph.

    """
    def __init__(self, losses, optimizer,
                 clip_gradients=5.0,
                 metrics=None,
                 batch_size=64,
                 batch_per_epoch=None,
                 trainable_vars=None,
                 step_tensor=None,
                 epoch_tensor=None,
                 tb_verbosity=0,
                 logdir=''):
        if not isinstance(losses, dict) or not all([isinstance(l, tf.Tensor) for l in losses.values()]):
            raise ValueError("Unknown Loss type")

        if not isinstance(metrics, dict) or not all([isinstance(t, tf.Tensor) for t in metrics.values()]) or not all([isinstance(k, str) for k in metrics.keys()]):
            raise ValueError("Metrics returned by model.evaluation_ops should be a dict of entries {metric_name: metric_tensor}")

        if not isinstance(optimizer, tf.train.Optimizer):
            raise ValueError("Unknown Optimizer")

        assert 0 <= tb_verbosity <= 3, 'Summary verbosity should be 0, 1 or 2.'

        self.graph = tf.get_default_graph()

        self.losses = losses
        self.losses_name = losses.keys()
        self.losses_value = {key: 0.0 for key in self.losses_name}
        self.metrics = metrics
        self.metrics_name = metrics.keys()
        self.metrics_value = {key: 0.0 for key in self.metrics_name}

        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch

        self.gradients = None
        self.train_vars = trainable_vars if trainable_vars is not None else tf.trainable_variables()
        all_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        assert all([var in self.train_vars for var in all_train_vars]), 'Uninitialized vars:\n' + '\n'.join([var.op.name  for var in all_train_vars if var not in self.train_vars])

        self.step_tensor = step_tensor
        self.epoch_tensor = epoch_tensor
        self.curr_step = 0
        self.curr_epoch = 0
        self.step_duration = []

        # Building training ops
        # Compute total loss, which is the loss of all optimizers plus the loss of all regularizers.
        with tf.variable_scope('losses'):
            all_losses = {'name': losses.keys(), 'tensor': losses.values()}
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if reg_loss:
                all_losses['name'].append('regularization')
                all_losses['tensor'].append(tf.add_n(reg_loss))
            total_loss = tf.add_n(all_losses['tensor'], name='total_loss')
            all_losses['name'].append('total_loss')
            all_losses['tensor'].append(total_loss)
        _add_scalar_summaries(all_losses['tensor'], all_losses['name'])

        # Compute and apply gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if not update_ops:
            update_ops = None

        with tf.variable_scope('optimizer'), tf.control_dependencies(update_ops):
            self.gradients = tf.gradients(total_loss, self.train_vars)
            if clip_gradients > 0.0:
                self.gradients, grad_norm = tf.clip_by_global_norm(self.gradients, clip_gradients)

            grads_and_vars = list(zip(self.gradients, self.train_vars))
            self.train_ops = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.step_tensor, name="apply_grad_op")

        # Summarize losses and metrics
        # Create other useful summaries (weights, grads, activations...)
        # with tf.variable_scope('metrics'):
        _add_scalar_summaries([self.metrics[m] for m in self.metrics_name], self.metrics_name)
        if '_lr_t' in optimizer.__dict__:
            _add_scalar_summaries([optimizer._lr_t], ['lr'])
        _add_histogram_summaries(self.train_vars, self.gradients, tb_verbosity)

        # Building summary ops
        self.summary_ops = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='TrainOp')+
                                            tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train_input'))
        self.summary_writer = tf.summary.FileWriter(logdir, graph=self.graph, flush_secs=30)

    def _print_stats(self):
        # Print stats to shell
        samples_per_sec = len(self.step_duration[-20:])*self.batch_size/sum(self.step_duration[-20:])
        min_per_epoch = sum(self.step_duration[-20:])/60.*self.batch_per_epoch/len(self.step_duration[-20:])
        timestamp = datetime.now()

        stats_str = '%s: TRAIN | step %d' % (timestamp, self.curr_step)
        stats_str += ' (%.1f examples/sec; %.3f min/epoch)' % (samples_per_sec, min_per_epoch)
        losses_str = '\n'.join(['%s: TRAIN | \t %s %.2f' % (timestamp, l, self.losses_value[l]) for l in self.losses_name])
        metrics_str = '\n'.join(['%s: TRAIN | \t %s %.2f' % (timestamp, m, self.metrics_value[m]) for m in self.metrics_name])

        print(stats_str)
        print(losses_str)
        print(metrics_str)
        sys.stdout.flush()

    def step(self, session, n=1):
        import time

        if False:
            import time
            print('Wait 2 mints to fill queue')
            time.sleep(120)

            from tensorflow.python.client import timeline
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            outputs = session.run([self.train_ops, self.epoch_tensor, self.step_tensor, self.summary_ops] +
                                  [self.losses[l] for l in self.losses_name] +
                                  [self.metrics[m] for m in self.metrics_name],
                                  options=run_options,
                                  run_metadata=run_metadata)

            tl = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.json', 'w') as f:
                f.write(tl.generate_chrome_trace_format())
            exit(0)

        for _ in range(n):
            # Train with one batch
            start_time = time.time()
            outputs = session.run([self.train_ops, self.epoch_tensor, self.step_tensor, self.summary_ops] +
                                  [self.losses[l] for l in self.losses_name] +
                                  [self.metrics[m] for m in self.metrics_name])
            self.step_duration.append(time.time() - start_time)

            # Update monitors
            self.curr_epoch, self.curr_step, summary_str, = outputs[1], outputs[2], outputs[3]
            for l, val in zip(self.losses_name, outputs[4:4+len(self.losses)]):
                self.losses_value[l] = val
            for m, val in zip(self.metrics_name, outputs[4+len(self.losses):]):
                self.metrics_value[m] = val
            assert not any(np.isnan(self.losses_value.values())), 'Model diverged with loss = NaN'

            # Manage stats and summaries
            if self.curr_step % 20 == 0 or self.curr_step == 1:
                self._print_stats()
            if self.curr_step % 100 == 0 or self.curr_step == 1:
                _write_summaries(summary_str, self.summary_writer, self.curr_step)


class EvaluateOp(object):
    """ TrainOp.

    TrainOp represents a set of operation used for optimizing a network.

    A TrainOp is meant to hold all training parameters of an optimizer.
    `Trainer` class will then instantiate them all specifically considering all
    optimizers of the network (set names, scopes... set optimization ops...).

    Arguments:
        loss: `Tensor`. Loss operation to evaluate network cost.
            Optimizer will use this cost function to train network.
        optimizer: `Optimizer`. Tensorflow Optimizer. The optimizer to
            use to train network.
        metric:  `Tensor`. The metric tensor to be used for evaluation.
        batch_size: `int`. Batch size for datalib feeded to this optimizer.
            Default: 64.
        ema: `float`. Exponential moving averages.
        trainable_vars: list of `tf.Variable`. List of trainable variables to
            use for training. Default: all trainable variables.
        shuffle: `bool`. Shuffle datalib.
        step_tensor: `tf.Tensor`. A variable holding training step. If not
            provided, it will be created. Early defining the step tensor
            might be useful for network creation, such as for learning rate
            decay.
        validation_monitors: `list` of `Tensor` objects.  List of variables
            to compute during validation, which are also used to produce
            summaries for output to TensorBoard.  For example, this can be
            used to periodically record a confusion matrix or AUC metric,
            during training.  Each variable should have rank 1, i.e.
            shape [None].
        validation_batch_size: `int` or None. If `int`, specifies the batch
            size to be used for the validation datalib feed; otherwise
            defaults to being th esame as `batch_size`.
        name: `str`. A name for this class (optional).
        graph: `tf.Graph`. Tensorflow Graph to use for training. Default:
            default tf graph.

    """
    def __init__(self, losses, metrics,
                 num_steps=50,
                 batch_size=64,
                 step_tensor=None,
                 epoch_tensor=None,
                 logdir=''):
        if not isinstance(losses, dict) or not all([isinstance(l, tf.Tensor) for l in losses.values()]):
            raise ValueError("Unknown Loss type")

        if not isinstance(metrics, dict) or not all([isinstance(t, tf.Tensor) for t in metrics.values()]) or not all([isinstance(k, str) for k in metrics.keys()]):
            raise ValueError("Metrics returned by model.evaluation_ops should be a dict of entries {metric_name: metric_tensor}")

        self.graph = tf.get_default_graph()

        self.losses = losses
        self.losses_name = losses.keys()
        self.losses_avg_value = {key: 0.0 for key in self.losses_name}
        self.metrics = metrics
        self.metrics_name = metrics.keys()
        self.metrics_avg_value = {key: 0.0 for key in self.metrics_name}

        self.batch_size = batch_size
        self.num_steps = num_steps

        self.step_tensor = step_tensor
        self.epoch_tensor = epoch_tensor
        self.curr_step = 0
        self.curr_epoch = 0
        self.duration = 0

        # Building validation ops
        with tf.variable_scope('losses'):
            self.loss_avg_tens = {l: tf.placeholder(tf.float32, shape=(), name=l) for l in self.losses_name}
        _add_scalar_summaries([self.loss_avg_tens[l] for l in self.losses_name], self.losses_name)

        with tf.variable_scope('metrics'):
            self.metrics_avg_tens = {m: tf.placeholder(tf.float32, shape=(), name=m) for m in self.metrics_name}
        _add_scalar_summaries([self.metrics_avg_tens[m] for m in self.metrics_name], self.metrics_name)

        # Building summary ops
        self.summary_ops = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='EvalOp'))
        self.summary_writer = tf.summary.FileWriter(logdir, graph=self.graph, flush_secs=30)

    def _print_stats(self):
        # Print stats to shell
        timestamp = datetime.now()
        stats_str = '%s: VALID | step %d' % (timestamp, self.curr_step)
        stats_str = ', '.join([stats_str] + ['%s = %.2f' % (l, self.losses_avg_value[l]) for l in self.losses_name])
        stats_str += ' (%.2f sec / %d examples)' % (self.duration, self.batch_size*self.num_steps)
        metrics_str = '\n'.join(['%s: VALID | \t %s %.2f' % (timestamp, m, self.metrics_avg_value[m]) for m in self.metrics_name])

        print('-'*70)
        print(stats_str)
        print(metrics_str)
        print('-'*70)
        sys.stdout.flush()

    def step(self, session):
        import time
        self.losses_avg_value, self.metrics_avg_value = {l: 0.0 for l in self.losses_name}, {m: 0.0 for m in self.metrics_name}
        output = session.run([self.epoch_tensor, self.step_tensor])
        self.curr_epoch, self.curr_step, self.duration = output[0], output[1], 0.0
        t = time.time()
        for _ in range(self.num_steps):
            output = session.run([self.losses[l] for l in self.losses_name] + [self.metrics[m] for m in self.metrics_name])
            for l, val in zip(self.losses_name, output[:len(self.losses)]):
                self.losses_avg_value[l] += val / self.num_steps
            for m, val in zip(self.metrics_name, output[len(self.losses):]):
                self.metrics_avg_value[m] += val/self.num_steps
        self.duration = time.time() - t

        # Run summary operation.
        feed_dict = {self.loss_avg_tens[l]: self.losses_avg_value[l] for l in self.losses_name}
        feed_dict.update({self.metrics_avg_tens[m]: self.metrics_avg_value[m] for m in self.metrics_name})
        test_summ_str = session.run(self.summary_ops, feed_dict=feed_dict)
        _write_summaries(test_summ_str, self.summary_writer, self.curr_step)
        self._print_stats()


def _add_scalar_summaries(tensor_list, tensor_names):
    if tensor_list:
        # Attach a scalar summary to all individual losses and metrics.
        for name, tensor in zip(tensor_names, tensor_list):
            tf.summary.scalar('%s' % name, tensor)


def _add_histogram_summaries(variables, gradients, verbosity=3):
    if verbosity > 0:
        # Add histograms for activation.
        activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
        actv_names = ['/'.join(actv.op.name.split('/')[1:]) for actv in activations]
        for name, act in zip(actv_names, activations):
            tf.summary.histogram('actv/'+name, act)

    if verbosity > 1:
        # Add histograms for variables.
        var_names = ['/'.join(var.op.name.split('/')[1:]) for var in variables]
        for name, var in zip(var_names, variables):
            tf.summary.histogram('var/'+name, var)

    if verbosity > 2:
        # Add histograms for gradients.
        for name, grad in zip(var_names, gradients):
            tf.summary.histogram('grad/'+name, grad)


def _write_summaries(summary_str, summary_writer, step):
    # Write summaries to Tensorboard
    summary_writer.add_summary(summary_str, step)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def test_dnn():
    import shutil
    from tflib import CustomModel, DNN, DataGenerator
    from tflib import wrappers as tfw

    class MyDataGenerator(DataGenerator):
        def __init__(self, num_samples=10000, num_classes=2, shuffle=True, batch_size=64):
            super(MyDataGenerator, self).__init__(shuffle, batch_size)
            self.add_input('X', 'float32', (10,))
            self.add_target('Y', 'int32', ())

            X, Y = [], []
            mean = np.zeros((2, 10))
            mean[0] = [0,1,1,0,1,0,1,0,0,1]
            mean[1] = [1,0,0,1,0,0,0,1,0,1]
            for c in xrange(num_classes):
                X.append(mean[c] + np.random.rand(num_samples, 10))
                Y.append(c*np.ones(num_samples, dtype=int))
            self.X = np.concatenate(X, axis=0)
            self.Y = np.concatenate(Y, axis=0)
            self.n_samples = self.Y.size

        def loop(self):
            from itertools import cycle
            order = np.arange(self.n_samples)
            if self.shuffle:
                np.random.shuffle(order)
            for item in cycle(order):
                yield {'X': self.X[item]}, {'Y': self.Y[item]}

        def batch_loop(self, num_epoch=np.inf):
            order = np.arange(self.n_samples)
            batch_lims = [(i, i+self.batch_size) for i in xrange(0, self.n_samples, self.batch_size)]

            n = 0
            while True:
                n += 1
                if n == num_epoch:
                    break
                if self.shuffle:
                    np.random.shuffle(order)
                for lims in batch_lims:
                    yield {'X': self.X[lims[0]:lims[1]]}, {'Y': self.Y[lims[0]:lims[1]]}

        def get_sample(self, index):
            return {'X': self.X[index]}, {'Y': self.Y[index]}

    class LogisticRegression(CustomModel):
        def __init__(self, n_classes):
            super(LogisticRegression, self).__init__()
            self.n_classes = n_classes

        def inference_ops(self, inputs, is_training=None):
            with tf.variable_scope('fc1'):
                x = tfw.fully_connected(inputs, 50, activation_fn=tf.nn.relu, is_training=is_training)
            with tf.variable_scope('logit'):
                return tfw.fully_connected(x, self.n_classes, activation_fn=None)

        def loss_ops(self, logits, targets):
            with tf.variable_scope('cross_entropy'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
                return tf.reduce_mean(loss, name='loss')

        def evaluation_ops(self, logits, targets, is_training=None):
            with tf.variable_scope('accuracy'):
                accuracy = tfw.metrics.accuracy(logits, targets)
            return {'accuracy': accuracy}

    train_dir = 'LogisticRegression'
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    train_data_gen = MyDataGenerator(num_classes=2, num_samples=10000, batch_size=64)
    val_data_gen = MyDataGenerator(num_classes=2, num_samples=1000, shuffle=False, batch_size=256)
    model = LogisticRegression(n_classes=2)

    dnn = DNN(tb_dir=train_dir, ckp_path=os.path.join(train_dir, 'model'), tb_verbosity=0)
    dnn.configure_session(gpu_fraction=0.1)
    dnn.training_setup(train_data_gen, model, batch_size=256, dt_valid=val_data_gen, num_feeders=2)
    dnn.train(n_epoch=10, snapshot_epoch=True, snapshot_step=20, validation_epoch=True, validation_step=-1)

    dnn = DNN(tb_dir=train_dir, ckp_path=os.path.join(train_dir, 'model'), tb_verbosity=0)
    dnn.deploy_setup(val_data_gen.input_info(), model, checkpoint=os.path.join(train_dir, 'model'))
    logits = dnn.predict(val_data_gen.X)
    print((logits.argmax(axis=1) == val_data_gen.Y).mean())


if __name__ == '__main__':
    test_dnn()
