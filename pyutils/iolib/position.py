import numpy as np
from pyutils.ambisonics.position import Position
from collections import OrderedDict


def read_position_file(fn):
    positions, wav_fns, img_fns = OrderedDict(), OrderedDict(), OrderedDict()
    sample_ids = []
    bg_img = None
    with open(fn, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            if line.startswith('<BGI>'):
                bg_img = line.split('<BGI>')[1]
                continue

            s = line.split()
            src_id = s[0]
            sample_ids.append(src_id)

            wav_fns[src_id] = s[1]
            if len(s) == 4:
                img_fns[src_id] = line.split()[2]

            num_pts = int(s[-1])
            positions[src_id] = []
            for _ in range(num_pts):
                p = [float(num) for num in f.readline().strip().split()]
                positions[src_id].append(Position(p[0], p[1], p[2], 'polar'))

    return sample_ids, positions, wav_fns, img_fns, bg_img


def save_position_fn(fn, source_ids, positions, source_wav, image_fns, bg_img=None):
    with open(fn, 'w') as f:
        if bg_img is not None:
            f.write('<BGI>{}<BGI>.\n'.format(bg_img))
        for src_id in source_ids:
            f.write('{} {} {} {}\n'.format(src_id, source_wav[src_id], image_fns[src_id], len(positions[src_id])))
            for p in positions[src_id]:
                f.write('{} {} {}\n'.format(p.phi, p.nu, p.r))


class PositionReader:
    def __init__(self, position_fn, org_dur, rate, pad_start=0, seek=None, duration=None, rotation=None):
        source_ids, positions, _, _, _ = read_position_file(position_fn)

        self.num_frames = int(org_dur * rate)
        self.positions = np.zeros((self.num_frames, 9))
        for idx, src_id in enumerate(source_ids):
            if len(positions[src_id]) == 1:
                pos = positions[src_id][0].coords('polar')
                pos = np.tile(pos[np.newaxis, :], (self.num_frames, 1))
            elif len(positions[src_id]) == 2:
                alpha = np.linspace(0, 1, self.num_frames)[:, np.newaxis]
                pos0 = positions[src_id][0].coords('polar')[np.newaxis, :]
                pos1 = positions[src_id][1].coords('polar')[np.newaxis, :]
                pos = alpha * pos1 + (1-alpha) * pos0
            elif len(positions[src_id]) == 0:
                continue
            else:
                raise ValueError('Too many points per source')
            self.positions[:, idx*3:(idx+1)*3] = pos

        if seek is not None:
            self.positions = self.positions[int(seek * rate):]
        if duration is not None:
            self.positions = self.positions[:int(duration * rate)]
        if rotation is not None:
            for i in range(len(source_ids)):
                self.positions[:, i * 3] += rotation
                idx = self.positions[:, i * 3] >= np.pi
                self.positions[idx, i * 3] -= 2 * np.pi
                idx = self.positions[:, i * 3] < -np.pi
                self.positions[idx, i * 3] += 2 * np.pi

        self.head = 0
        self.pad = pad_start
        self.num_channels = self.positions.shape[1]


    def get_next_chunk(self, n=1, force_size=False):
        if self.head >= self.num_frames:
            return None

        frames_left = self.num_frames - self.head
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
        chunk = self.positions[self.head:self.head+chunk_size]
        self.head += chunk_size

        if pad_chunk is not None:
            chunk = np.concatenate((pad_chunk, chunk), axis=0)
        return chunk

    def loop_chunks(self, n=1, force_size=False):
        while True:
            snippet = self.get_next_chunk(n, force_size=force_size)
            if snippet is None:
                break
            yield snippet
