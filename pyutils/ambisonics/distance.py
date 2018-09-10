import numpy as np
from common import AmbiFormat
from decoder import AmbiDecoder
from pyutils.iolib.position import read_position_file
from pyutils.ambisonics.position import Position, MovingSource
from itertools import izip


def spherical_mesh(angular_res):
    phi_rg = np.flip(np.arange(-180., 180., angular_res) / 180. * np.pi, 0)
    nu_rg = np.arange(-90., 90.1, angular_res) / 180. * np.pi
    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)
    return phi_mesh, nu_mesh


class SphericalAmbisonicsVisualizer(object):
    def __init__(self, data, rate=22050, window=0.1, angular_res=2.0):
        self.window = window
        self.angular_res = angular_res
        self.data = data
        self.phi_mesh, self.nu_mesh = spherical_mesh(angular_res)
        mesh_p = [Position(phi, nu, 1., 'polar') for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))]

        # Setup decoder
        ambi_order = np.sqrt(data.shape[1]) - 1
        self.decoder = AmbiDecoder(mesh_p, AmbiFormat(ambi_order=ambi_order, sample_rate=rate), method='projection')

        # Compute spherical energy averaged over consecutive chunks of "window" secs
        self.window_frames = int(self.window * rate)
        self.n_frames = data.shape[0] / self.window_frames
        self.output_rate = float(rate)/self.window_frames
        self.frame_dims = self.phi_mesh.shape
        self.cur_frame = -1

    def visualization_rate(self):
        return self.output_rate

    def mesh(self):
        return self.nu_mesh, self.phi_mesh

    def get_next_frame(self):
        self.cur_frame += 1
        if self.cur_frame >= self.n_frames:
            return None

        # Decode ambisonics on a grid of speakers
        chunk_ambi = self.data[self.cur_frame * self.window_frames:((self.cur_frame + 1) * self.window_frames), :]
        decoded = self.decoder.decode(chunk_ambi)

        # Compute RMS at each speaker
        rms = np.sqrt(np.mean(decoded ** 2, 0)).reshape(self.phi_mesh.shape)
        return np.flipud(rms)

    def loop_frames(self):
        while True:
            rms = self.get_next_frame()
            if rms is None:
                break
            yield rms


class SphericalSourceVisualizer(object):
    def __init__(self, position_fn, duration, rate=10., angular_res=5):
        from sklearn.neighbors import KDTree
        phi_mesh, nu_mesh = spherical_mesh(angular_res)

        x_mesh = np.cos(nu_mesh) * np.cos(phi_mesh)
        y_mesh = np.cos(nu_mesh) * np.sin(phi_mesh)
        z_mesh = np.sin(nu_mesh)
        self.p_mesh = np.stack((x_mesh, y_mesh, z_mesh), 0).reshape((3, -1))
        self.kdtree = KDTree(self.p_mesh.T, leaf_size=2, metric='euclidean')
        self.nframes = int(duration*rate)
        self.frame_dims = phi_mesh.shape

        positions, _, source_ids = read_position_file(position_fn)
        self.sources = [MovingSource(np.zeros((self.nframes,)), positions[src_id], rate) for src_id in source_ids]
        self.cur_frame = -1

    def get_next_frame(self):
        self.cur_frame += 1
        if any([not src.tic() for src in self.sources]):
            return None

        pmap = np.zeros((self.frame_dims[0]*self.frame_dims[1],))
        for src in self.sources:
            p_cart = src.position.coords('cartesian').reshape((1, 3))
            opt = self.kdtree.query(p_cart, return_distance=False)
            pmap[opt] += 1./len(self.sources)

        return pmap.reshape(self.frame_dims)

    def loop_frames(self):
        while True:
            pmap = self.get_next_frame()
            if pmap is None:
                break
            yield pmap


def emd(map1, map2, phi_mesh, nu_mesh):
    import pyemd
    x_mesh = np.cos(nu_mesh) * np.cos(phi_mesh)
    y_mesh = np.cos(nu_mesh) * np.sin(phi_mesh)
    z_mesh = np.sin(nu_mesh)
    p_mesh = np.stack((x_mesh, y_mesh, z_mesh), 0).reshape((3, -1))
    ang_dist = np.dot(p_mesh.T, p_mesh)
    ang_dist[ang_dist >= 1] = 1
    ang_dist[ang_dist <= -1] = -1
    ang_dist = np.arccos(ang_dist)

    if map1.ndim == 2:
        map1 = map1[np.newaxis, :, :]
    if map2.ndim == 2:
        map2 = map2[np.newaxis, :, :]

    # Average over time domain (1st dimension)
    nframes = map1.shape[0]
    map1 = map1.reshape((nframes, -1))
    map2 = map2.reshape((nframes, -1))
    emd_dist = np.zeros((nframes,))
    emd_dist2 = np.zeros((nframes,))
    n_nodes = map1[0].size
    for t in range(nframes):
        emd_dist[t] = pyemd.emd(map1[t] / n_nodes, map2[t] / n_nodes, ang_dist)
        emd_dist2[t] = pyemd.emd(map1[t] / (map1[t].sum()+0.01), map2[t] / (map2[t].sum()+0.01), ang_dist)
    return emd_dist.mean(), emd_dist2.mean() #((c1 - c2) ** 2).mean()


def ambix_emd(ambi1, ambi2, rate, ang_res=20):
    # Computation time with pyemd
    # Mesh: full connectivity
    # Angular Resolution | No. nodes | Computation time
    #             20 deg |      162  | ~0.5sec / second of audio
    #             15 deg |      288  | ~5sec   / second of audio
    #             10 deg |      648  | ~100sec / second of audio

    ambi1Vis = SphericalAmbisonicsVisualizer(ambi1, rate, window=0.1, angular_res=ang_res)
    ambi2Vis = SphericalAmbisonicsVisualizer(ambi2, rate, window=0.1, angular_res=ang_res)
    directional_error, power_error = [], []
    for rms1, rms2 in izip(ambi1Vis.loop_frames(), ambi2Vis.loop_frames()):
        derr, perr = emd(rms1, rms2, ambi1Vis.phi_mesh, ambi1Vis.nu_mesh)
        directional_error.append(derr), power_error.append(perr)
    return np.mean(directional_error), np.mean(power_error)


def test_emd():
    from pyutils.iolib.audio import load_wav

    # Load ambisonics
    ang_res = 10
    sample = 'wav_test/gen_synthetic-M1'
    data, rate = load_wav(sample+'-ambix.wav')
    duration = data.shape[0]/float(rate)

    ambiVis = SphericalAmbisonicsVisualizer(data, rate, window=0.1, angular_res=ang_res)
    # vid_reader = VideoReader(sample+'.avi', ambiVis.visualization_rate(),
    #                          image_preprocessing=lambda x: resize(rgb2gray(x), ambiVis.phi_mesh.shape))
    srcVis = SphericalSourceVisualizer(sample+'-position.txt',
                                       duration,
                                       rate=ambiVis.visualization_rate(),
                                       angular_res=ang_res)

    for rms, frame in izip(ambiVis.loop_frames(), srcVis.loop_frames()):
        print emd(rms, frame, ambiVis.phi_mesh, ambiVis.nu_mesh)
        # plt.imshow(frame + rms/rms.max())
        # plt.show()


def test_ambix_emd():
    from pyutils.iolib.audio import load_wav
    # Run from project home (spatialaudiogen/)

    # Load ambisonics
    rate = 24000
    ambi1, _ = load_wav('data/wav_test/hello-left2right-ambix.wav', rate=rate)
    ambi2, _ = load_wav('data/wav_test/hello-statright-ambix.wav', rate=rate)
    print 'Same FOA: EMD =', ambix_emd(ambi1, ambi1, rate)
    print 'Diff FOA: EMD =', ambix_emd(ambi1, ambi2, rate)

if __name__ == '__main__':
    # test_emd()
    test_ambix_emd()
