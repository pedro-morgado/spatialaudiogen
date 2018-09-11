import os, sys
import numpy as np
from pyutils.cmd import runSystemCMD
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, '3rd-party/vrProjector/')
import vrProjector

def dir2samples(path):
    files = os.listdir(path)
    files = [fn for fn in files if len(fn.split('.')) and fn.split('.')[-1] in ('webm', 'mp4', 'mkv', 'm4a')]
    files = [fn for fn in files if fn != 'downloaded.txt']
    youtube_ids = [fn.split('.')[0] for fn in files]

    samples = {yid: [] for yid in youtube_ids}
    for yid, fn in zip(youtube_ids, files):
        samples[yid].append(os.path.join(path, fn))
    return samples, samples.keys()


def nonZeroChannels(inp_fn):
    from scipy.io import wavfile
    sndfile = '/tmp/output.wav'
    cmd = 'ffmpeg -y -t 300 -i {} -map a -ar 10000 {}'.format(inp_fn, sndfile)
    out, stderr = runSystemCMD(cmd)
    if any([l.startswith('Output file is empty') for l in stderr.split('\n')]):
        raise ValueError('ERROR: Output file is empty\nCMD: {}\n STDERR: {}'.format(cmd, stderr))

    fs, data = wavfile.read(sndfile)
    os.remove(sndfile)
    return (data != 0).sum(axis=0) > 0


def extract_clip(inp_fn, out_fn, rate=10, seek=None, duration=None):
    cmd = ['ffmpeg', '-y']
    if seek is not None:
        cmd += ['-ss', '{0:.10f}'.format(seek)]
    cmd += ['-i', inp_fn]
    if duration is not None:
        cmd += ['-t', '{0:.10f}'.format(duration)]
    cmd += ['-an',
            '-vf', 'scale=720:360',
            '-r', str(rate),
            '-vcodec', 'libx264',
            '-crf', '5',
            out_fn]

    stdout, stderr = runSystemCMD(' '.join(cmd))
    if any([l.startswith('Output file is empty,')
            for l in stderr.split('\n')]):
        raise ValueError, 'Output file is empty.\n' + stderr


def my_interp2(data, x, y, pts):
    if data.ndim == 2:
        interp = RegularGridInterpolator((y, x), data, bounds_error=False, method='linear')
        return interp(pts)
    elif data.ndim == 3:
        out = []
        for i in range(data.shape[2]):
            interp = RegularGridInterpolator((y, x), data[:, :, i], bounds_error=False, method='linear')
            out.append(interp(pts))
        return np.stack(out, axis=-1)


def my_imresize(data, shape):
    dshape = data.shape
    assert data.ndim >= len(shape)

    dx = np.arange(dshape[1])/float(dshape[1])
    dy = np.arange(dshape[0])/float(dshape[0])

    dx_grid_n, dy_grid_n = np.meshgrid(range(shape[1]), range(shape[0]))
    dx_grid_n, dy_grid_n = dx_grid_n/float(shape[1]), dy_grid_n/float(shape[0])
    pts = np.stack((dy_grid_n.reshape(-1), dx_grid_n.reshape(-1)), axis=1)

    if len(dshape) == 2:
        data = data.reshape(dshape+(1,))
    if len(dshape) > 3:
        data = data.reshape(dshape[:2]+np.prod(dshape[2:]))

    outp = my_interp2(data, dx, dy, pts)

    if len(dshape) == 2:
        outp = outp[:, :, 0]
    else:
        outp = outp.reshape(shape[:2]+dshape[2:])
    return outp


def unwarp_eac(face):
    dims = face.shape
    x = np.arange(dims[1])/float(dims[1]-1)-0.5
    y = np.arange(dims[0])/float(dims[0]-1)-0.5

    x_grid, y_grid = np.meshgrid(x, y)
    x_new = np.arctan(2*x_grid)*2/np.pi
    y_new = np.arctan(2*y_grid)*2/np.pi

    pts = np.stack((y_new.reshape(-1), x_new.reshape(-1)), axis=1)
    new_img = my_interp2(face, x, y, pts).reshape(dims)
    new_img[np.isnan(new_img)] = 0
    return new_img


def cub2eqr(left, front, right, top, back, bottom, width=720, height=360, dtype=np.uint8):
    source = vrProjector.CubemapProjection()
    source.setImages(front, right, back, left, top, bottom)
    source.set_use_bilinear(True)
    out = vrProjector.EquirectangularProjection()
    out.initImage(width, height, dtype=dtype)
    out.reprojectToThis(source)
    return out.image


def gen_eac2eqr_maps(eac_shape, eqr_shape, stereopsis='MONO'):
    # Input grids
    xgrid, ygrid = np.meshgrid(range(eac_shape[1]), range(eac_shape[0]))
    eac_grid = np.stack((xgrid, ygrid), axis=2)

    # Grab 1st stereo channel only
    if stereopsis == 'STEREO':
        eac_grid = np.rot90(eac_grid[:, :eac_shape[1]/2, :], -1)

    # Split faces
    hs = eac_grid.shape[0]/2
    ws = eac_grid.shape[1]/3
    face_dims = (min(hs, ws), min(hs, ws))
    faces = ['left', 'front', 'right', 'top', 'back', 'bottom']
    eac_grid = {'left': my_imresize(eac_grid[:hs, :ws, :], face_dims),
                'front': my_imresize(eac_grid[:hs, ws:2*ws, :], face_dims),
                'right': my_imresize(eac_grid[:hs, 2*ws:, :], face_dims),
                'bottom': my_imresize(np.rot90(eac_grid[hs:, :ws, :], -1), face_dims),
                'back': my_imresize(np.rot90(eac_grid[hs:, ws:2*ws, :]), face_dims),
                'top': my_imresize(np.rot90(eac_grid[hs:, 2*ws:, :], -1), face_dims)}

    # EAC to CubeMap
    cub_grid = {f: unwarp_eac(eac_grid[f]) for f in faces}
    cub_grid = {f: np.pad(cub_grid[f], ((0, 0), (0, 0), (0, 1)), 'constant') for f in faces}

    # CubeMap to EquiRect
    eqr_grid = cub2eqr(width=eqr_shape[1], height=eqr_shape[0], dtype=np.float32, **cub_grid)
    xmap, ymap = eqr_grid[:, :, 0], eqr_grid[:, :, 1]
    return xmap, ymap

def save_pgm(fp, map, mmax):
    height, width = map.shape[:2]
    fp.write('P2\n{} {}\n{}\n'.format(width, height, mmax))
    for i in range(height):
        fp.write(' '.join([str(num) for num in map[i, :]])+'\n')
