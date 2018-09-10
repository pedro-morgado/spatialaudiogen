import os
import random
import threading
import sys
import numpy as np
import skimage.io as sio
import tensorflow as tf
from definitions import *
from pyutils.iolib.audio import load_wav


class FilenameProvider(object):
    def __init__(self, directory,
                 subset_fn=None,
                 num_epochs=1,
                 shuffle=False):
        self.directory = directory
        self.sample_ids = os.listdir(directory)
        assert len(self.sample_ids) > 0, 'Dataset directory is empty.'

        if subset_fn is not None:
            assert os.path.exists(subset_fn)
            subset = open(subset_fn).read().splitlines()
            self.sample_ids = [y for y in self.sample_ids if y in subset]

        self.num_epochs, self.epoch = num_epochs, 0
        self.num_samples = len(self.sample_ids)
        self.shuffle = shuffle
        self.head = -1

    def get_next_sample(self):
        self.head = (self.head + 1) % self.num_samples
        if self.head == 0:
            self.epoch += 1
            if self.epoch > self.num_epochs:
                return None
            if self.shuffle:
                random.shuffle(self.sample_ids)

        return self.sample_ids[self.head]

    def loop_samples(self):
        while True:
            yid = self.get_next_sample()
            if yid is None:
                break
            yield yid


class AudioReader(object):
    def __init__(self, audio_folder, rate=None, ambi_order=1):
        from scikits.audiolab import Sndfile
        self.audio_folder = audio_folder

        fns = os.listdir(audio_folder)
        self.num_files = len(fns)

        fp = Sndfile(os.path.join(self.audio_folder, fns[0]), 'r')
        self.rate = float(fp.samplerate) if rate is None else rate
        self.num_channels = min((fp.channels, (ambi_order+1)**2))
        self.duration = self.num_files
        self.num_frames = int(self.duration * rate)

    def get(self, start_time, size, rotation=None):
        # Check if padding is necessary
        start_frame = int(start_time * self.rate)
        pad_before, pad_after = 0, 0
        if start_frame < 0:
            pad_before = abs(start_frame)
            size -= pad_before
            start_time, start_frame = 0., 0
        if start_frame + size > self.num_frames:
            pad_after = start_frame + size - self.num_frames
            size -= pad_after
            
        # Load audio
        index = range(int(start_time), min(int(np.ceil(start_time + size / float(self.rate))), self.num_files))
        fns = ['{}/{:06d}.wav'.format(self.audio_folder, i) for i in index]
        chunk = [load_wav(fn, self.rate)[0] for fn in fns]
        chunk = np.concatenate(chunk, axis=0) if len(chunk) > 1 else chunk[0]
        ss = int((start_time - int(start_time)) * self.rate)
        chunk = chunk[ss:ss + size, :self.num_channels]

        # Pad
        if pad_before > 0:
            pad = np.zeros((pad_before, self.num_channels))
            chunk = np.concatenate((pad, chunk), axis=0)
        if pad_after > 0:
            pad = np.zeros((pad_after, self.num_channels))
            chunk = np.concatenate((chunk, pad), axis=0)

        # Apply rotation
        if rotation is not None:
            assert -np.pi <= rotation < np.pi
            c = np.cos(rotation)
            s = np.sin(rotation)
            rot_mtx = np.array([[1, 0, 0, 0],  # W' = W
                                [0, c, 0, s],  # Y' = X sin + Y cos
                                [0, 0, 1, 0],  # Z' = Z
                                [0, -s, 0, c]]) # X' = X cos - Y sin
            chunk = np.dot(chunk, rot_mtx.T)

        return chunk


class VideoReader(object):
    def __init__(self, video_folder, rate=None, img_prep=None):
        raw_rate = 10.
        self.video_folder = video_folder
        self.rate = rate if rate is not None else raw_rate
        self.img_prep = img_prep if img_prep is not None else lambda x: x

        frame_fns = [fn for fn in os.listdir(video_folder) if fn.endswith('.jpg')]
        self.num_frames = len(frame_fns)
        self.duration = self.num_frames / raw_rate

        img = sio.imread(os.path.join(video_folder, frame_fns[0]))
        self.frame_shape = self.img_prep(img).shape

    def get_by_index(self, start_time, size, rotation=None):
        ss = max(int(start_time * self.rate), 0)

        chunk = []
        for fno in range(ss, ss+size):
            fn = os.path.join(self.video_folder, '{:06d}.jpg'.format(fno))
            frame = self.img_prep(sio.imread(fn))
            chunk.append(frame)
        chunk = np.stack(chunk, 0) if len(chunk) > 1 else chunk[0][np.newaxis]
        if rotation is not None:
            roll = -int(rotation / (2. * np.pi) * self.frame_shape[1])
            chunk = np.roll(chunk, roll, axis=2)
        return chunk


class FlowReader(object):
    def __init__(self, flow_dir, flow_lims_fn, rate=None, flow_prep=None):
        self.reader = VideoReader(flow_dir, rate=rate)
        self.lims = np.load(flow_lims_fn)
        self.rate = self.reader.rate
        self.duration = self.reader.duration
        self.flow_prep = flow_prep if flow_prep is not None else lambda x: x

        dummy_img = self.flow_prep(np.zeros(self.reader.frame_shape[:2], dtype=np.float32))
        self.frame_shape = dummy_img.shape + (1,)
        self.dtype = dummy_img.dtype

    def get_by_index(self, start_time, size, rotation=None):
        chunk = self.reader.get_by_index(start_time, size, rotation)
        chunk = chunk.astype(np.float32)

        ss = max(int(start_time * self.rate), 0)
        t = chunk.shape[0]
        m_min = self.lims[ss:ss+t, 0].reshape((-1, 1, 1))
        m_max = self.lims[ss:ss+t, 1].reshape((-1, 1, 1))

        chunk[:, :, :, 2] *= (m_max - m_min) / 255.
        chunk[:, :, :, 2] += m_min
        chunk[:, :, :, 0] *= (2 * np.pi) / 255.
        chunk[:, :, :, 1] = chunk[:, :, :, 2] * np.sin(chunk[:, :, :, 0])
        chunk[:, :, :, 0] = chunk[:, :, :, 2] * np.cos(chunk[:, :, :, 0])
        return chunk


class SampleReader(object):
    """ Sample reader that preprocesses one sample (ambisonics, video)."""
    def __init__(self, folder,
                 ambi_order=1,
                 audio_rate=48000,
                 video_rate=10,
                 context=1.0,
                 duration=0.1,
                 return_video=True,
                 img_prep=None,
                 return_flow=False,
                 flow_prep=None,
                 skip_silence_thr=None,
                 shuffle=True,
                 start_time=0.5,
                 sample_duration=None,
                 skip_rate=None,
                 random_rotations=True,
                 num_threads=1,
                 thread_id=0):
        a2v = float(audio_rate) / video_rate
        snd_dur = duration * audio_rate
        vid_dur = duration * video_rate
        snd_ctx = context * audio_rate

        self.video_id = os.path.split(folder)[-1]

        # Check input settings
        assert a2v==int(a2v)
        assert float(snd_dur)==int(snd_dur)
        assert float(vid_dur)==int(vid_dur)
        assert float(snd_ctx)==int(snd_ctx)

        # Readers
        self.audio_reader = AudioReader(os.path.join(folder, 'ambix'), audio_rate, ambi_order)
        self.video_reader = VideoReader(os.path.join(folder, 'video'), video_rate, img_prep)
        if return_flow:
            flow_dir = os.path.join(folder, 'flow')
            flow_lims = os.path.join(folder, 'flow', 'flow_limits.npy')
            self.flow_reader = FlowReader(flow_dir, flow_lims, video_rate, flow_prep)

        # Store arguments
        self.folder = folder
        self.duration = duration
        self.context = context
        self.audio_rate = audio_rate
        self.video_rate = video_rate
        self.audio_size = int(snd_dur) + int(snd_ctx) - 1
        self.video_size = int(vid_dur)
        self.video_shape = self.video_reader.frame_shape
        self.return_video = return_video
        self.return_flow = return_flow
        self.random_rotations = random_rotations

        # If is not training, iterate through video, else extract random time frames
        audio_pow_fn = os.path.join(folder, 'audio_pow.lst')
        chunks_t = [float(l.strip().split()[0]) for l in open(audio_pow_fn)]
        chunks_pow = [float(l.strip().split()[1]) for l in open(audio_pow_fn)]
        if skip_rate is not None:
            num_chunks = len(chunks_t)
            chunks_t = [chunks_t[i] for i in range(0, num_chunks, skip_rate)]
            chunks_pow = [chunks_pow[i] for i in range(0, num_chunks, skip_rate)]
        if skip_silence_thr is not None:
            chunks_t = [chunks_t[i] for i in range(len(chunks_t)) if chunks_pow[i]>skip_silence_thr]
        if start_time > 0.5:
            chunks_t = [chunks_t[i] for i in range(len(chunks_t)) if chunks_t[i]>=start_time]
        if sample_duration is not None:
            chunks_t = [chunks_t[i] for i in range(len(chunks_t)) if chunks_t[i]<start_time+sample_duration]
        if num_threads > 1:
            lims = np.linspace(0, len(chunks_t), num_threads+1).astype(int)
            chunks_t = chunks_t[lims[thread_id]:lims[thread_id+1]]
        if shuffle:
            random.shuffle(chunks_t)
        self.chunks_t = chunks_t
        self.head = -1


    def get(self):
        self.head += 1
        if self.head >= len(self.chunks_t):
            return None
        self.cur_t = self.chunks_t[self.head]
        cur_t = self.cur_t
        rotation = random.random() * 2 * np.pi - np.pi if self.random_rotations else None
        chunks = {'id': self.video_id + ' ' + str(cur_t)}

        # Audio
        audio_ss = cur_t - self.context / 2
        chunks['ambix'] = self.audio_reader.get(audio_ss, self.audio_size, rotation)
        assert chunks['ambix'] is not None, 'Could not get ambix data for file {} (sec: {})'.format(self.folder, audio_ss)

        # Video
        if self.return_video:
            chunks['video'] = self.video_reader.get_by_index(cur_t, self.video_size, rotation)
            assert chunks['video'] is not None, 'Could not get video data for file {} (frame: {})'.format(self.folder, cur_t)

        # Flow
        if self.return_flow:
            chunks['flow'] = self.flow_reader.get_by_index(cur_t, self.video_size, rotation)
            assert chunks['flow'] is not None, 'Could not get flow data for file {} (frame: {})'.format(self.folder, cur_t)

        return chunks

    def loop_chunks(self, n=np.inf):
        k = 0
        while True:
            k += 1
            if k > n:
                break

            chunks = self.get()
            if chunks is None:
                break
            else:
                yield chunks


class Feeder(object):
    """ Background feeder that preprocesses audio and video files
        and enqueues them into a TensorFlow queue."""
    def __init__(self, sample_dir,
                 subset_fn=None,
                 ambi_order=1,
                 audio_rate=48000,
                 video_rate=10,
                 context=1.0,
                 duration=0.1,
                 return_video=True,
                 frame_size=None,
                 img_prep=None,
                 return_flow=False,
                 flow_prep=None,
                 queue_size=32,
                 n_threads=1,
                 for_eval=False):

        self.sample_dir, self.subset_fn = sample_dir, subset_fn
        self.ambi_order = ambi_order
        self.audio_rate, self.video_rate = audio_rate, video_rate
        self.context, self.duration = context, duration
        self.return_video = return_video
        self.img_prep = img_prep
        self.return_flow = return_flow
        self.flow_prep = flow_prep
        self.n_threads, self.threads = n_threads, []
        self.for_eval = for_eval
        self.skip_silence_thr = None if for_eval else (0.01 if 'REC-Street' in self.subset_fn else 0.2)

        audio_layouts = 'meta/audio_layouts.txt'
        masks = {'WXYZ': np.array([1., 1., 1., 1.]), 'WXY': np.array([1., 1., 0., 1.])}
        self.channel_mask = {l.split()[0]: masks[l.split()[1]] for l in open(audio_layouts).read().splitlines()}

        # Placeholders
        snd_ctx = int(context * audio_rate)
        snd_dur = int(duration * audio_rate)
        snd_shape = (snd_dur + snd_ctx - 1, int(ambi_order+1) ** 2)
        vid_dur = int(duration * video_rate)
        vid_shape = (vid_dur, frame_size[0], frame_size[1], 3)
        
        names = ['id', 'ambix', 'audio_mask']
        shapes = [(), snd_shape, ((self.ambi_order+1)**2,)]
        dtypes = [tf.string, tf.float32, tf.float32]
        if return_video:
            names += ['video']
            shapes += [vid_shape]
            dtypes += [tf.float32]
        if return_flow:
            names += ['flow']
            shapes += [vid_shape]
            dtypes += [tf.float32]

        self.tba = {m: tf.placeholder(dtype=t, shape=s) for m, s, t in zip(names, shapes, dtypes)}

        # Setup tf queue
        self.queue = tf.PaddingFIFOQueue(queue_size, names=names, dtypes=dtypes, shapes=shapes)
        self.enqueue = self.queue.enqueue(self.tba)
        self.queue_state = self.queue.size()

        # Print feeder state
        fn_provider = FilenameProvider(self.sample_dir, subset_fn=self.subset_fn, num_epochs=1)
        n_chunks = 0
        for yid in fn_provider.loop_samples():
            folder = os.path.join(self.sample_dir, yid)
            reader = SampleReader(folder, skip_silence_thr=self.skip_silence_thr,
                                  skip_rate=10 if self.for_eval else None)
            n_chunks += len(reader.chunks_t)
        print('\n'+'='*20, 'Feeder', '='*20)
        print('{:20s} | {}'.format('Input directory', fn_provider.directory))
        print('{:20s} | {}'.format('# videos', fn_provider.num_samples))
        print('{:20s} | {}'.format('# chunks', n_chunks))
        print('{:20s} | {}'.format('# threads', self.n_threads))
        print('{:20s} | {}'.format('Mode', 'eval' if self.for_eval else 'train'))

        print('{:20s} | {}'.format('Video fps', video_rate))
        print('{:20s} | {} frames, {} secs'.format('Video context', 0, 0))
        print('{:20s} | {} frames, {} secs'.format('Video duration', vid_dur, duration))
        print('{:20s} | {}'.format('Audio rate', audio_rate))
        print('{:20s} | {} frames, {} secs'.format('Audio context', snd_ctx, context))
        print('{:20s} | {} frames, {} secs'.format('Audio duration', snd_dur, duration))

        print('\nFeeder output tensors')
        for m, s, t in zip(names, shapes, dtypes):
            print(' * {:10s} | {:20s} | {:10s}'.format(m, str(s), str(t)))
        sys.stdout.flush()

    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)

    def thread_main(self, sess, thread_id, num_threads):
        thread = threading.currentThread()
        fn_provider = FilenameProvider(self.sample_dir, subset_fn=self.subset_fn,
                                       num_epochs=1 if self.for_eval else np.inf,
                                       shuffle=not self.for_eval)
        
        NUM_SAMPLING = np.inf if self.for_eval else 5
        SKIP_RATE = 10 if self.for_eval else None
        thread_id = thread_id if self.for_eval else 0
        num_threads = num_threads if self.for_eval else 1
        for yid in fn_provider.loop_samples():
            # Start readers
            folder = os.path.join(self.sample_dir, yid)
            reader = SampleReader(folder,
                                  ambi_order=self.ambi_order,
                                  audio_rate=self.audio_rate,
                                  video_rate=self.video_rate,
                                  context=self.context,
                                  duration=self.duration,
                                  return_video=self.return_video,
                                  img_prep=self.img_prep,
                                  return_flow=self.return_flow,
                                  flow_prep=self.flow_prep,
                                  skip_silence_thr=self.skip_silence_thr,
                                  shuffle=not self.for_eval,
                                  random_rotations=not self.for_eval,
                                  skip_rate=SKIP_RATE,
                                  thread_id=thread_id,
                                  num_threads=num_threads)

            # Feed data into tf queue
            for chunk in reader.loop_chunks(NUM_SAMPLING):
                feed_dict = {self.tba[n]: chunk[n] for n in chunk}
                feed_dict[self.tba['audio_mask']] = self.channel_mask[yid]

                if not thread.should_stop:
                    sess.run(self.enqueue, feed_dict=feed_dict)
                else:
                    return

    def done(self, sess):
        for t in self.threads:
            if t.isAlive():
                return False
        qsize = sess.run(self.queue_state)
        if qsize >= 32:
            return False
        return True 

    def start_threads(self, sess):
        # Launch feeding threads
        for i in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess, i, self.n_threads))
            thread.should_stop = False
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads

    def join(self):
        for t in self.threads:
            t.should_stop = True
            t.join()
