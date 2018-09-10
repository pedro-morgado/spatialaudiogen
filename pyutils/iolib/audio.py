import os
import scipy.signal
import numpy as np
from pyutils.iolib.video import getFFprobeMeta
from pyutils.cmd import runSystemCMD
from scikits.audiolab import Sndfile, Format
import tempfile
import resampy


def load_wav(fname, rate=None):
    fp = Sndfile(fname, 'r')
    _signal = fp.read_frames(fp.nframes)
    _signal = _signal.reshape((-1, fp.channels))
    _rate = fp.samplerate

    if _signal.ndim == 1:
        _signal.reshape((-1, 1))
    if rate is not None and rate != _rate:
        # _num_frames = _signal.shape[0]
        # _duration = _num_frames / float(_rate)
        # signal = scipy.signal.resample(_signal, int(rate * _duration))
        signal = resampy.resample(_signal, _rate, rate, axis=0, filter='kaiser_fast')
    else:
        signal = _signal
        rate = _rate

    return signal, rate


def save_wav(fname, signal, rate):
    fp = Sndfile(fname, 'w', Format('wav'), signal.shape[1], rate)
    fp.write_frames(signal)
    fp.close()


def convert2wav(inp_fn, out_fn, rate=None):
    cmd = ['ffmpeg', '-y',
           '-i', inp_fn,
           '-map', '0:a',
           '-acodec', 'pcm_s16le']
    if rate is not None:
        cmd += ['-ar', str(rate),]
    cmd += [out_fn]

    stdout, stderr = runSystemCMD(' '.join(cmd))
    if any([l.startswith('Output file is empty,')
            for l in stderr.split('\n')]):
        raise ValueError, 'Output file is empty.\n' + stderr


class AudioReader:
    def __init__(self, fn, rate=None, pad_start=0, seek=None, duration=None, rotation=None):
        fp = Sndfile(fn, 'r') if fn.endswith('.wav') else None
        if fp is None or (rate is not None and fp.samplerate != rate):
            # Convert to wav file
            if not os.path.isdir('/tmp/'):
                os.makedirs('/tmp/')
            snd_file = tempfile.NamedTemporaryFile('w', prefix='/tmp/', suffix='.wav', delete=False)
            snd_file.close()

            convert2wav(fn, snd_file.name, rate)
            self.snd_fn = snd_file.name
            self.rm_flag = True

        else:
            self.snd_fn = fn
            self.rm_flag = False

        self.fp = Sndfile(self.snd_fn, 'r')
        self.num_channels = self.fp.channels
        self.rate = self.fp.samplerate
        self.num_frames = self.fp.nframes
        self.duration = self.num_frames / float(self.rate)

        self.k = 0
        self.pad = pad_start

        if seek is not None and seek > 0:
            num_frames = int(seek * self.rate)
            self.fp.read_frames(num_frames)
        else:
            seek = 0

        if duration is not None:
            self.duration = min(duration, self.duration-seek)
            self.num_frames = int(self.duration * self.rate)

        if rotation is not None:
            assert self.num_channels > 2    # Spatial audio
            assert -np.pi <= rotation < np.pi
            c = np.cos(rotation)
            s = np.sin(rotation)
            rot_mtx = np.array([[1, 0, 0, 0],       # W' = W
                                [0, c, 0, s],       # Y' = X sin + Y cos
                                [0, 0, 1, 0],       # Z' = Z
                                [0, -s, 0, c]])     # X' = X cos - Y sin
            self.rot_mtx = rot_mtx
        else:
            self.rot_mtx = None

    def __del__(self):
        if self.rm_flag:
            os.remove(self.snd_fn)

    def get_chunk(self, n=1, force_size=False):
        if self.k >= self.num_frames:
            return None

        frames_left = self.num_frames - self.k
        if force_size and n > frames_left:
            return None

        # Pad zeros to start
        if self.pad > 0:
            pad_size = min(n, self.pad)
            pad_chunk = np.zeros((pad_size, self.num_channels))
            n -= pad_size
            self.pad -= pad_size
        else:
            pad_chunk = None

        # Read frames
        chunk_size = min(n, frames_left)
        chunk = self.fp.read_frames(chunk_size)
        chunk = chunk.reshape((chunk.shape[0], self.num_channels))
        self.k += chunk_size

        if pad_chunk is not None:
            chunk = np.concatenate((pad_chunk.astype(chunk.dtype), chunk), 0)

        if self.rot_mtx is not None:
            chunk = np.dot(chunk, self.rot_mtx.T)

        return chunk

    def loop_chunks(self, n=1, force_size=False):
        while True:
            chunk = self.get_chunk(n, force_size=False)
            if chunk is None:
                break
            yield chunk

class AudioReader2:
    def __init__(self, audio_folder, rate=None,
                 seek=0, duration=None, rotation=None):
        self.audio_folder = audio_folder

        fns = os.listdir(audio_folder)
        self.num_files = len(fns)

        fp = Sndfile(os.path.join(self.audio_folder, fns[0]), 'r')
        data, fps = load_wav(os.path.join(self.audio_folder, fns[0]))
        self.rate = float(fp.samplerate) if rate is not None else fps
        self.num_channels = fp.channels
        self.duration = self.num_files
        self.num_frames = int(self.duration * rate)

        self.cur_frame = int(seek * self.rate)
        self.time = self.cur_frame / self.rate

        self.max_time = self.duration
        if duration is not None:
            self.max_time = min(seek + duration, self.max_time)

        if rotation is not None:
            assert self.num_channels > 2  # Spatial audio
            assert -np.pi <= rotation < np.pi
            c = np.cos(rotation)
            s = np.sin(rotation)
            rot_mtx = np.array([[1, 0, 0, 0],  # W' = W
                                [0, c, 0, s],  # Y' = X sin + Y cos
                                [0, 0, 1, 0],  # Z' = Z
                                [0, -s, 0, c]])  # X' = X cos - Y sin
            self.rot_mtx = rot_mtx
        else:
            self.rot_mtx = None

    def get(self, start_time, size):
        index = range(int(start_time), int(start_time + size / self.rate) + 1)
        fns = [os.path.join(self.audio_folder, '{:06d}.wav'.format(i))
               for i in index]
        chunk = []
        for fn in fns:
            if not os.path.exists(fn):
                return None
            data, _ = load_wav(fn, self.rate)
            chunk.append(data)

        chunk = np.concatenate(chunk, 0) if len(chunk) > 1 else chunk[0]
        ss = int((start_time - int(start_time)) * self.rate)
        chunk = chunk[ss:ss+size, :]

        return chunk

    def get_chunk(self, n=1, force_size=False):
        if self.time >= self.max_time:
            return None

        frames_left = int((self.max_time - self.time) * self.rate)
        if force_size and n > frames_left:
            return None

        # Read frames
        chunk_size = min(n, frames_left)
        start_time = self.cur_frame / self.rate
        end_frame_no = self.cur_frame + chunk_size - 1
        end_time = end_frame_no / self.rate

        index = range(int(start_time), int(end_time) + 1)
        fns = [os.path.join(self.audio_folder, '{:06d}.wav'.format(i))
               for i in index]
        chunk = []
        for fn in fns:
            data, _ = load_wav(fn, self.rate)
            chunk.append(data)
        chunk = np.concatenate(chunk, 0) if len(chunk) > 1 else chunk[0]
        ss = int((self.time - int(self.time)) * self.rate)
        chunk = chunk[ss:ss+chunk_size, :]
        self.cur_frame += chunk.shape[0]
        self.time = self.cur_frame / self.rate

        if self.rot_mtx is not None:
            chunk = np.dot(chunk, self.rot_mtx.T)

        return chunk

    def loop_chunks(self, n=1, force_size=False):
        while True:
            chunk = self.get_chunk(n, force_size=False)
            if chunk is None:
                break
            yield chunk


def test_audio_reader():
    reader = AudioReader2('/gpu2_data/morgado/spatialaudiogen/youtube/train/687gkvLi5kI/ambix',
                         rate=10000, seek=0, duration=5.5)
    for s in reader.loop_chunks(10000):
        print(s.shape), s.max(), s.min()
# test_audio_reader()

