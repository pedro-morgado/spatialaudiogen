import os, sys, tempfile, glob, argparse
sys.path.insert(0, '.')
import shutil
import numpy as np
from skimage import io as sio
from pyutils.iolib.video import getFFprobeMeta
from utils import gen_eac2eqr_maps, save_pgm
from pyutils.iolib.audio import save_wav, AudioReader
from pyutils.iolib.video import VideoReader
import feeder
import multiprocessing as mp


def prepare_ambisonics(inp_fn, out_fn, inp_codec, overwrite=False):
    if overwrite and os.path.exists(out_fn):
        os.remove(out_fn)
    if not overwrite and os.path.exists(out_fn):
        return
    assert not os.path.exists(out_fn)

    cmd = 'ffmpeg -y -i "{}" -vn -ar 48000'.format(inp_fn)
    if inp_codec == 'aac':
        remap = [2, 1, 4, 0]
    elif inp_codec in ('vorbis', 'opus'):
        remap = [0, 1, 2, 3]
    else:
        raise ValueError, '{}: Unkown input codec: {}.'.format(inp_fn, inp_codec)
    cmd += ' -af "pan=4c|c0=c{}|c1=c{}|c2=c{}|c3=c{}"'.format(*remap)
    cmd += ' "{}"'.format(out_fn)
    
    print("\nINPUT:", inp_fn, inp_codec)
    print("COMMAND:", cmd)
    print("")
    stdout = os.popen(cmd).read()
    

def prepare_video(inp_fn, stereopsis, projection, out_fn, out_shape, out_rate, overwrite=False):
    if overwrite and os.path.exists(out_fn):
        os.remove(out_fn)
    if not overwrite and os.path.exists(out_fn):
        return
    assert not os.path.exists(out_fn)

    meta = getFFprobeMeta(inp_fn)['video']
    height, width = int(meta['height']), int(meta['width'])

    inputs = [inp_fn]
    filter_chain = []
    if projection == 'ER':
        # Split stereo if necessary and scale down
        if stereopsis == 'STEREO':
            filter_chain.append('crop=in_w:in_h/2:0:0')
        filter_chain.append('scale={}:{}'.format(out_shape[1], out_shape[0]))

    elif projection == 'EAC':
        xmap_fn = 'scraping/pgms/xmap_{}x{}_{}x{}_{}.pgm'.format(height, width, out_shape[0]*2, out_shape[1]*2, stereopsis)
        ymap_fn = 'scraping/pgms/ymap_{}x{}_{}x{}_{}.pgm'.format(height, width, out_shape[0]*2, out_shape[1]*2, stereopsis)
        if not os.path.isfile(xmap_fn) or not os.path.isfile(ymap_fn):
            # Generate coord maps
            xmap, ymap = gen_eac2eqr_maps((height, width), (out_shape[0]*2, out_shape[1]*2), stereopsis)

            # Save coord maps
            with open(xmap_fn, 'w') as f:
                save_pgm(f, xmap.astype(np.uint16), 2**16-1)

            with open(ymap_fn, 'w') as f:
                save_pgm(f, ymap.astype(np.uint16), 2**16-1)

        inputs.extend([xmap_fn, ymap_fn])

    # Run ffmpeg
    cmd = ['ffmpeg -y -ss 0']
    for inp in inputs:
        cmd += ['-i', '"'+inp+'"']
    cmd += ['-an', '-r', str(out_rate)]  # 10, 30
    if projection == 'EAC':
        cmd += ['-lavfi', 'remap']
    else:
        # Remap+pix_fmt turns image into gray-scale ????
        cmd += ['-pix_fmt', 'yuv420p']

    if filter_chain:
        cmd += ['-vf', ','.join(filter_chain)]
    cmd += ['"'+out_fn+'"']

    print(' '.join(cmd))
    stdout = os.popen(' '.join(cmd)).read()

    # Clean pgm files
    if projection == 'EAC':
        tmp_fn = tempfile.NamedTemporaryFile(prefix='/tmp/', suffix='.mp4', delete=False)
        tmp_fn.close()
        os.system('mv {} {}'.format(out_fn, tmp_fn.name))
        stdout = os.popen('ffmpeg -i "{}" -pix_fmt yuv420p -vf scale={}:{} "{}"'.format(tmp_fn.name, out_shape[1], out_shape[0], out_fn)).read()
        os.remove(tmp_fn.name)


def extract_frames(audio_fn, video_fn, frames_dir, yid, overwrite):
    print('\n'+'='*30+' '+yid+' '+'='*30)

    # Prepare directory tree
    if not os.path.isdir(frames_dir):
        os.makedirs(frames_dir)

    audio_dir = os.path.join(frames_dir, 'ambix')
    if os.path.isdir(audio_dir):
        if overwrite:
            shutil.rmtree(audio_dir)
            os.makedirs(audio_dir)
    else:
        os.makedirs(audio_dir)

    video_dir = os.path.join(frames_dir, 'video')
    if os.path.isdir(video_dir):
        if overwrite:
            shutil.rmtree(video_dir)
            os.makedirs(video_dir)
    else:
        os.makedirs(video_dir)

    # Open readers
    audio_reader = AudioReader(audio_fn)
    video_reader = VideoReader(video_fn)
    duration_secs = int(min(audio_reader.duration, video_reader.duration))

    # Ambisonics
    print('({}) Splitting ambisonics into chunks'.format(yid))
    sys.stdout.flush()
    reader = AudioReader(audio_fn, rate=48000)
    for i in range(duration_secs):
        chunk_fn = os.path.join(audio_dir, '{:06d}.wav'.format(i))
        chunk = reader.get_chunk(reader.rate)
        save_wav(chunk_fn, chunk, reader.rate)

    # Video
    print('({}) Splitting video into frames'.format(yid))
    sys.stdout.flush()
    reader = VideoReader(video_fn)
    num_frames = int(reader.fps * duration_secs)
    for i in range(num_frames):
        frame_fn = os.path.join(video_dir, '{:06d}.jpg'.format(i))
        img = reader.get()
        sio.imsave(frame_fn, img)


def compute_audio_pow(audio_dir, output_fn):
    reader = feeder.AudioReader(audio_dir, rate=48000)
    with open(output_fn, 'w') as f:
        for i in range((reader.duration-1)*10):
            t = i/10.+0.5
            signal = reader.get(t, 4800)
            apow = np.sqrt((signal[:, 0]**2).mean(axis=0))
            f.write('{} {}\n'.format(t, apow))


def compute_flow(video_dir, flow_dir, flownet2_dir, gpu=0):
    from flow import FlowNet2

    if os.path.isdir(flow_dir):
        shutil.rmtree(flow_dir)
    os.makedirs(flow_dir)
    

    fn = os.path.join(flow_dir, 'flow_limits.npy')

    video_fns = os.listdir(video_dir)
    video_frames = [int(fn.split('.')[0]) for fn in video_fns]
    video_fns = [os.path.join(video_dir, video_fns[i]) for i in np.argsort(video_frames).tolist()]
    prev_img = sio.imread(video_fns[0])

    height, width = prev_img.shape[:2]
    caffemodel = os.path.join(flownet2_dir, 'models', 'FlowNet2', 'FlowNet2_weights.caffemodel.h5')
    deployproto = os.path.join(flownet2_dir, 'models', 'FlowNet2', 'FlowNet2_deploy.prototxt.template')
    flow_machine = FlowNet2(height, width, caffemodel, deployproto, gpu=gpu)

    magnitude_lims = []
    for i, fn in enumerate(video_fns):
        out_fn = os.path.join(flow_dir, '{:06d}.jpg'.format(i))
        next_img = sio.imread(fn)
        flow = flow_machine.compute(prev_img, next_img)
        prev_img = next_img

        # Process frame for compressed storage
        mag = np.sqrt((flow ** 2).sum(axis=2))
        ang = np.arctan2(flow[:, :, 1], flow[:, :, 0]) + np.pi
        ang[mag < 0.005] = 0

        m_min, m_max = mag.min(), mag.max()
        if m_max - m_min < 1:  # Avoid 0 division
            m_max = m_min + 1

        magnitude_lims.append([m_min, m_max])
        flow_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        flow_rgb[..., 0] = (ang * 255. / (np.pi * 2.))
        flow_rgb[..., 1] = 0
        flow_rgb[..., 2] = (mag - m_min) / (m_max - m_min) * 255.

        # Save frame
        sio.imsave(out_fn, flow_rgb)
        if i % 100 == 0:
            print('{}/{}'.format(i+1, len(video_fns)))

    fn = os.path.join(flow_dir, 'flow_limits.npy')
    np.save(fn, np.array(magnitude_lims))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_list', default='meta/spatialaudiogen_db.lst', help='File containing list of youtube ids.')
    parser.add_argument('--orig_dir', default='data/orig', help='Folder containing downloaded videos.')
    parser.add_argument('--output_prep_dir', default='data/preproc', help='Output folder for preprocessed videos.')
    parser.add_argument('--output_frames_dir', default='data/frames', help='Output folder for preprocessed video frames and audio chunks.')
    parser.add_argument('--output_prep_hr_dir', default='data/preproc-hr', help='Output folder for preprocessed high-resolution videos.')
    parser.add_argument('--flownet2_dir', default='', help='Path to flownet2 older for preprocessed high-resolution videos.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of parallel workers.')
    parser.add_argument('--prep_hr_video', action='store_true', help='Flag to pre-process videos in high-resolution (for deployment only).')
    parser.add_argument('--low_res', action='store_true', help='Flag if low-resolution videos have been downloaded.')
    parser.add_argument('--gpus', default='0', type=str, help='GPU ids available. (eg 0,1,2)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    if args.low_res:
        assert not args.prep_hr_video

    if not os.path.isdir(args.output_prep_dir):
        os.makedirs(args.output_prep_dir)
    if args.prep_hr_video and not os.path.isdir(args.output_prep_hr_dir):
        os.makedirs(args.output_prep_hr_dir)
    if not os.path.isdir(args.output_frames_dir):
        os.makedirs(args.output_frames_dir)
    if not os.path.isdir('scraping/pgms'):
        os.makedirs('scraping/pgms')

    def worker(q, gpu):
        while not q.empty():
            yid, orig_audio_fn, orig_video_fn = q.get()
            print('='*10, int(q.qsize()), 'remaining', yid, '='*10)

            # Prepare audio
            audio_formats_fn = 'scraping/audio_formats.txt'
            audioenc = [l.split()[2] for l in open(audio_formats_fn) if l.split()[0]==yid][0]
            prep_audio_fn = os.path.join(args.output_prep_dir, '{}-ambix.m4a'.format(yid))
            prepare_ambisonics(orig_audio_fn, prep_audio_fn, audioenc, args.overwrite)

            # Prepare video
            video_formats_fn = 'scraping/video_formats_lowres.txt' if args.low_res else 'scraping/video_formats.txt'
            stereopsis = [l.split()[2] for l in open(video_formats_fn) if l.split()[0]==yid][0]
            projection = [l.split()[3] for l in open(video_formats_fn) if l.split()[0]==yid][0]
            prep_video_fn = os.path.join(args.output_prep_dir, '{}-video.mp4'.format(yid))
            prepare_video(orig_video_fn, stereopsis,  projection, prep_video_fn, (224, 448), 10, args.overwrite)
            if args.prep_hr_video:
                prep_hr_video_fn = os.path.join(args.output_prep_hr_dir, '{}-video.mp4'.format(yid))
                prepare_video(orig_video_fn, stereopsis,  projection, prep_hr_video_fn, (1080, 1920), 30, args.overwrite)
                
            # Extract frames for training
            frames = os.path.join(args.output_frames_dir, yid)
            extract_frames(prep_audio_fn, prep_video_fn, frames, yid, args.overwrite)
            
            # Precompute audio volume (silence is ignored during training)
            compute_audio_pow(os.path.join(frames, 'ambix'), os.path.join(frames, 'audio_pow.lst'))

            # Precompute flow (requires instalation of flownet2)
            if args.flownet2_dir:
                compute_flow(os.path.join(frames, 'video'), os.path.join(frames, 'flow'), args.flownet2_dir, gpu)
            break


    to_process = open(sys.argv[1]).read().splitlines()
    audio_files = {os.path.split(fn)[-1].split('.')[0]: fn for fn in glob.glob('{}/*.audio.*'.format(args.orig_dir))}
    video_files = {os.path.split(fn)[-1].split('.')[0]: fn for fn in glob.glob('{}/*.video.*'.format(args.orig_dir))}
    q = mp.Queue()
    for yid in to_process:
        if yid in audio_files and yid in video_files:
            q.put((yid, audio_files[yid], video_files[yid]))

    proc = []
    gpus_avail = args.gpus.split(',')
    for i in range(args.num_workers):
        gpu = int(gpus_avail[i%len(gpus_avail)])
        p = mp.Process(target=worker, args=(q, gpu))
        p.daemon = True
        p.start()
        proc.append(p)

    for p in proc:
        p.join()
