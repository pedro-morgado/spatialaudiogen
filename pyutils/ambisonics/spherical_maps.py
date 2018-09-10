import numpy as np
from common import AmbiFormat
from decoder import AmbiDecoder
from pyutils.ambisonics.position import Position


def spherical_mesh(angular_res):
    phi_rg = np.flip(np.arange(-180., 180., angular_res) / 180. * np.pi, 0)
    nu_rg = np.arange(-90., 90., angular_res) / 180. * np.pi
    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)
    return phi_mesh, nu_mesh


class SphericalMapMachine(object):
    def __init__(self, ambi_order=1, window=None, angular_res=20.0):
        self.angular_res = angular_res
        self.phi_mesh, self.nu_mesh = spherical_mesh(angular_res)
        self.frame_shape = self.phi_mesh.shape
        self.window = window
        mesh_p = [Position(phi, nu, 1., 'polar')
                  for phi, nu in zip(self.phi_mesh.reshape(-1),
                                     self.nu_mesh.reshape(-1))]

        # Setup decoder
        self.decoder = AmbiDecoder(mesh_p, AmbiFormat(ambi_order), method='projection')

    def compute(self, data):
        # Decode ambisonics on a grid of speakers
        if self.window is not None:
            n_windows = data.shape[0] / self.window
            data = data[:self.window*n_windows]
        decoded = self.decoder.decode(data)

        # Compute RMS at each speaker
        if self.window is not None:
            decoded = decoded.reshape((n_windows, self.window, -1))
            rms = np.sqrt(np.mean(decoded ** 2, 1))
            rms = rms.reshape((n_windows,) + self.frame_shape)
        else:
            rms = np.sqrt(np.mean(decoded ** 2, 0))
            rms = rms.reshape(self.frame_shape)

        return rms
