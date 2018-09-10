import os, re
import numpy as np
from pyutils.cmd import runSystemCMD
import skimage.io as sio

OPENCV = 0
IMAGEIO = 1
FFMPEG = 2
BACKENDS = {'opencv': OPENCV, 'imageio': IMAGEIO, 'ffmpeg': FFMPEG}


def getFFprobeMeta(fn):
    cmd = 'ffprobe -hide_banner -loglevel panic ' + fn + ' -show_streams'
    log, _ = runSystemCMD(cmd)

    log = log.split('\n')
    start_stream = [i for i, l in enumerate(log) if l == '[STREAM]']
    end_stream = [i for i, l in enumerate(log) if l == '[/STREAM]']

    streams = []
    for init, end in zip(start_stream, end_stream):
        streams.append(log[init+1:end])

    meta = dict()
    for stream in streams:
        tmp = {}
        for l in stream:
            m = re.match('(.*)=(.*)', l)
            if m is None:
                continue
            tmp[m.group(1)] = m.group(2)
        meta[tmp['codec_type']] = tmp
    return meta


class BasicVideoReader:
    def __init__(self, video_fn,
                 backend='imageio',
                 fps=None,
                 seek=0,
                 duration=None):
        self.backend = BACKENDS[backend]
        if self.backend == IMAGEIO:
            import imageio
            self.reader = imageio.get_reader(video_fn)#, format='AVBIN')
            self.reader_iter = self.reader.iter_data()

            w, h = self.reader._meta['size']
            self.frame_shape = (h, w)

            self._raw_duration = self.reader._meta['duration']
            self._raw_fps = self.reader._meta['fps']
            self._raw_frames = int(self._raw_duration * self._raw_fps)

        elif self.backend == OPENCV:
            import cv2

            self.reader = cv2.VideoCapture(video_fn)

            w = int(self.reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
            h = int(self.reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            self.frame_shape = (h, w)

            self._raw_frames = int(self.reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            self._raw_fps = self.reader.get(cv2.cv.CV_CAP_PROP_FPS)
            self._raw_duration = self._raw_frames / float(self._raw_fps)

        else:
            raise ValueError, 'Backend not supported'

        if duration is not None:
            self.end_time = min(duration + seek, self._raw_duration)
        else:
            self.end_time = self._raw_duration
        self.duration = self.end_time - seek
        self.fps = fps if fps is not None else self._raw_fps

        self._raw_delta = 1. / self._raw_fps
        self._raw_frame_no = -1
        self._raw_time = self._raw_frame_no * self._raw_delta
        self.delta = 1. / self.fps
        self.frame_no = -1
        self.time = self.frame_no * self.delta
        self.eof = False
        
        if seek > 0:
            n_skips = int(seek * self.fps)
            self.skip(n_skips)

    def _read_next(self):
        if self.eof:
            return None

        self._raw_frame_no += 1
        self._raw_time = self._raw_frame_no * self._raw_delta
        if self._raw_time > self.end_time:
            self.eof = True
            return None

        if self.backend == IMAGEIO:
            try:
                frame = next(self.reader_iter)
            except StopIteration:
                self.eof = True

        elif self.backend == OPENCV:
            success, frame = self.reader.read()
            if not success:
                self.eof = True

        return frame if not self.eof else None

    def get(self):
        if self.eof:
            return None

        EPS = 1e-4
        self.frame_no += 1
        self.time = self.frame_no * self.delta
        while self._raw_time < self.time - EPS and not self.eof:
            frame = self._read_next()

        if self.eof:
            self.frame_no -= 1
            self.time = self.frame_no * self.delta
            return None

        if self.backend == OPENCV:
            frame = np.flip(frame, 2)

        return frame

    def loop(self):
        while True:
            frame = self.get()
            if frame is None:
                break
            yield frame

    def skip(self, n):
        for _ in range(n):
            frame = self.get()
            if frame is None:
                return False
        return True


class FrameReader:
    def __init__(self, video_folder,
                 fps=None,
                 seek=0,
                 duration=None):
        self.video_folder = video_folder
        self.raw_fps = 10.

        frame_fns = os.listdir(video_folder)
        self.num_frames = len(frame_fns)
        self.duration = self.num_frames / self.raw_fps

        img = sio.imread(os.path.join(video_folder, frame_fns[0]))
        self.frame_shape = img.shape[:2]

        self.fps = fps if fps is not None else self.raw_fps
        self.delta = 1. / self.fps
        self.cur_frame = int(seek * self.fps) - 1
        self.time = self.cur_frame * self.delta

        self.max_frame = self.num_frames
        if duration is not None:
            self.max_frame = min(int((seek + duration) * self.fps), self.max_frame)

    def get_by_index(self, start_time, size):
        ss = int(start_time * self.fps)

        chunk = []
        for fno in range(ss, ss+size):
            fn = os.path.join(self.video_folder, '{:06d}.jpg'.format(fno))
            if not os.path.exists(fn):
                return None
            chunk.append(sio.imread(fn))
        return np.stack(chunk, 0) if len(chunk) > 1 else chunk[0][np.newaxis]

    def get(self):
        self.cur_frame += 1
        if self.cur_frame >= self.max_frame:
            return None

        self.time = self.cur_frame * self.delta
        frame_no = int(self.time * self.raw_fps)
        fn = os.path.join(self.video_folder, '{:06d}.jpg'.format(frame_no))
        return sio.imread(fn)

    def loop(self):
        while True:
            frame = self.get()
            if frame is None:
                break
            yield frame

    def skip(self, n):
        for _ in range(n):
            frame = self.get()
            if frame is None:
                return False
        return True


class VideoReader:
    def __init__(self, video_fn,
                 backend='imageio',
                 rate=None,
                 seek=0,
                 pad_start=0,
                 duration=None,
                 rotation=None,
                 img_prep=None):
        if pad_start != 0:
            assert seek == 0 and isinstance(pad_start, int)

        self.backend = backend
        if backend != 'jpg':
            self.reader = BasicVideoReader(video_fn, backend, rate, seek, duration)
        else:
            self.reader = FrameReader(video_fn, rate, seek, duration)

        self.fps = self.reader.fps
        self.duration = self.reader.duration
        self.num_frames = int (self.duration * self.fps)
        self.img_prep = img_prep if img_prep is not None else lambda x: x

        raw_shape = self.reader.frame_shape + (3, )
        dummy_img = self.img_prep(np.zeros(raw_shape, dtype=np.uint8))
        self.frame_shape = dummy_img.shape
        self.pad_start = pad_start

        if rotation is not None:
            assert -np.pi <= rotation < np.pi
            self.roll = -int(rotation / (2. * np.pi) * self.frame_shape[1])
        else:
            self.roll = None

    def get_by_index(self, start_time, size):
        assert self.backend == 'jpg'
        chunk = self.reader.get_by_index(start_time, size)
        if chunk is None:
            return None
        elif len(chunk) > 1:
            chunk = np.stack([self.img_prep(frame) for frame in chunk], 0)
        else:
            chunk = self.img_prep(chunk[0])[np.newaxis]
        return chunk

    def get(self):
        if self.pad_start > 0:
            frame = np.zeros(self.frame_shape, dtype=np.uint8)
            self.pad_start -= 1
        else:
            frame = self.reader.get()

        if frame is not None:
            frame = self.img_prep(frame)
            if self.roll is not None:
                frame = np.roll(frame, self.roll, axis=1)
        return frame

    def loop(self):
        while True:
            frame = self.get()
            if frame is None:
                break
            yield frame

    def get_chunk(self, n=1, force_size=False):
        chunk = []
        for i in range(n):
            frame = self.get()
            if frame is None:
                break

            chunk.append(frame)

        if len(chunk) == 0: # End of video
            return None

        if force_size and len(chunk) != n:  # Not have enough frames
            return None

        if len(chunk) > 1:
            return np.stack(chunk, axis=0)
        else:
            return np.expand_dims(chunk[0], 0)

    def loop_chunks(self, n=1, force_size=False):
        while True:
            chunk = self.get_chunk(n, force_size=force_size)
            if chunk is None:
                break
            yield chunk


class VideoWriter:
    def __init__(self, video_fn, video_fps, backend='imageio', codec='libx264', quality=5, overwrite=False):
        if overwrite and os.path.exists(video_fn):
            os.remove(video_fn)
        assert not os.path.exists(video_fn)

        self.backend = BACKENDS[backend]
        self.video_fn = video_fn
        self.fps = video_fps
        if self.backend == IMAGEIO:
            import imageio
            self.writer = imageio.get_writer(video_fn,
                                             fps=video_fps,
                                             codec=codec,
                                             pixelformat='yuv420p',
                                             quality=quality)

        else:
            raise ValueError, 'Backend not supported'

    def __del__(self):
        if self.backend == IMAGEIO:
            self.writer.close()

    def close(self):
        if self.backend == IMAGEIO:
            self.writer.close()

    def write_frame(self, frame):
        if self.backend == IMAGEIO:
            self.writer.append_data(frame)

    def write_chunk(self, chunk):
        for frame in chunk:
            self.write_frame(frame)


def test_basic_reader():
    import time
    from matplotlib import pyplot as plt
    fn = '../../data/youtube/preproc/3n0JIuX9fZA-video.mp4'
    reader = BasicVideoReader(fn, backend='opencv', seek=30, fps=10, duration=30)

    duration = []
    start_time = time.time()
    for v in reader.loop():
        duration.append(time.time() - start_time)
        print(reader.time, v.shape, duration[-1])
        plt.imshow(v)
        plt.show()
        start_time = time.time()

    print 'Done'
    print reader.time
# test_basic_reader()


def test_video_reader():
    from matplotlib import pyplot as plt
    import time
    fn = '/gpu2_data/morgado/spatialaudiogen/youtube/train/74ZiZ1iGG4k/video'
    reader = VideoReader(fn, backend='jpg', rate=10, pad_start=0, seek=0,
                         duration=10, rotation=np.pi/2, img_prep=None)
    duration = []
    start_time = time.time()
    for v in reader.loop_chunks(10):
        duration.append(time.time() - start_time)
        print(reader.reader.time, v.shape, duration[-1])
        plt.imshow(v[0])
        plt.show()
        start_time = time.time()
# test_video_reader()


def test_video_writer():
    import os
    fn = '../../data/youtube/preproc/3n0JIuX9fZA-video.mp4'
    reader = VideoReader(fn, rate=10, pad_start=0, seek=30,
                         duration=30, rotation=np.pi/2, img_prep=None)
    writer = VideoWriter('tmp.mp4', 10, backend='imageio', overwrite=True)
    for v in reader.loop():
        writer.write_frame(v)
        print(v.shape)
    writer.close()

    os.system('ffplay tmp.mp4')
    os.remove('tmp.mp4')
# test_video_writer()

