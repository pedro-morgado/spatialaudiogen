import numpy as np
from video import VideoReader


class FlowReader:
    def __init__(self, flow_video_fn, flow_lims_fn,
                 rate=None,
                 pad_start=0,
                 seek=None,
                 duration=None,
                 rotation=None,
                 flow_prep=None):
        self.reader = VideoReader(flow_video_fn,
                                  rate=rate,
                                  pad_start=pad_start,
                                  seek=seek,
                                  duration=duration,
                                  rotation=rotation)
        self.lims = np.load(flow_lims_fn)
        self.fps = self.reader.fps
        self.duration = self.reader.duration

        # Seek and store first frame
        if seek is not None:
            n = int(seek * self.reader._ifps)
            self.lims = self.lims[n:]

        if flow_prep is None:
            flow_prep = lambda x: x
        self.flow_prep = flow_prep

        dummy_img = self.flow_prep(np.zeros(self.reader.frame_shape[:2], dtype=np.float32))
        self.frame_shape = dummy_img.shape + (1,)
        self.dtype = dummy_img.dtype

    def get_next_frame(self):
        while True:
            flow = self.reader.get_next_frame()
            if flow is None:
                return None

            # Recover angle and magnitude
            cur_frame = int(self.reader._it * self.reader._ifps)
            m_min, m_max = self.lims[cur_frame, 0], self.lims[cur_frame, 1]
            magnt = flow[:, :, 2] / 255. * (m_max - m_min) + m_min
            return self.flow_prep(magnt)[:, :, np.newaxis]

            angle = flow[:, :, 0] / 255. * (2 * np.pi)
            x_flow = self.flow_prep(magnt * np.cos(angle))
            y_flow = self.flow_prep(magnt * np.sin(angle))
            return np.stack((x_flow, y_flow), axis=2)

    def loop_frames(self):
        while True:
            frame = self.get_next_frame()
            if frame is None:
                break
            yield frame

    def get_next_chunk(self, n=1, force_size=False):
        chunk = []
        for i in range(n):
            frame = self.get_next_frame()
            if frame is None:
                break

            chunk.append(frame)

        if len(chunk) == 0:
            return None

        if force_size and len(chunk) != n:
            return None

        return np.stack(chunk, axis=0) if len(chunk) > 1 else np.expand_dims(chunk[0], 0)

    def loop_chunks(self, n=1, force_size=False):
        while True:
            snippet = self.get_next_chunk(n, force_size=force_size)
            if snippet is None:
                break
            yield snippet


def test_reader():
    from matplotlib import pyplot as plt
    video_fn = '/mnt/ilcompf5d0/user/owang/data/spatialaudiogen/preproc/2YTzoWLAzA4-video.mp4'
    flow_fn = '/mnt/ilcompf5d0/user/owang/data/spatialaudiogen/preproc/2YTzoWLAzA4-flow.mp4'
    lims_fn = '/mnt/ilcompf5d0/user/owang/data/spatialaudiogen/preproc/2YTzoWLAzA4-flow.npy'
    RATE = 5
    SEEK = 30
    DURATION = 10

    flow_reader = FlowReader(flow_fn, lims_fn, rate=RATE, seek=SEEK, duration=DURATION)
    vid_reader = VideoReader(video_fn, rate=RATE, seek=SEEK, duration=DURATION)
    f, ax = plt.subplots(3, 1)
    while True:
        rgb_frame = vid_reader.get_next_frame()
        flow_frame = flow_reader.get_next_frame()
        if flow_frame is None and rgb_frame is None:
            break

        flow_frame = (flow_frame/25.*255).astype(np.uint8)
        print flow_frame.min(), flow_frame.max()

        ax[0].imshow(flow_frame, cmap='gray', vmin = 0, vmax = 255)
        ax[1].imshow(rgb_frame)
        ax[2].imshow(rgb_frame * (flow_frame[:, :, np.newaxis]>25).astype(np.uint8))
        plt.pause(0.1)

if __name__ == '__main__':
    test_reader()
