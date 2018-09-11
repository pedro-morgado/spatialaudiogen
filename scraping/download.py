import sys
import os
import argparse
import time
import re
import numpy as np


def select_video_formats(youtube_ids):
    done = [l.split()[0] for l in open('scraping/download_formats.txt')]
    with open('scraping/download_formats.txt', 'a+') as f:
        for ii, yid in enumerate(youtube_ids):
            if yid in done:
                continue
            url = 'https://www.youtube.com/watch?v=' + yid
            print '{}/{}: {}'.format(ii, len(youtube_ids), url)
                
            try:
                data_fmt = os.popen('youtube-dl -F "{}"'.format(url)).read().splitlines()
                data_fmt = [l.split() for l in data_fmt]
                data_fmt = [l for l in data_fmt if l[0].isdigit() and l[1] == 'mp4']
                width = np.array([float(l[2].split('x')[0]) for l in data_fmt])
                width[width<450] = np.inf
                video_fmt = data_fmt[np.argmin(width)][0]
                print yid, ' '.join(data_fmt[np.argmin(width)])
                f.write('{} {}\n'.format(yid, video_fmt))
            except Exception:
                print 'Error downloading', yid
                open('scraping/download_errors.txt', 'a').write(yid+'\n')


def download_video_with_fmt(url, fmt, video_dir):
    cmd = ['youtube-dl', '--ignore-errors', 
           '--download-archive', 'scraping/downloaded_video.txt', 
           '--format', fmt, 
           '-o', '"{}/%(id)s.video.%(ext)s"'.format(video_dir),
           '"{}"'.format(url)]
    os.system(' '.join(cmd))


def download_video(url, video_dir):
    cmd = ['youtube-dl', '--ignore-errors', 
           '--download-archive', 'scraping/downloaded_video.txt', 
           '--format', 'bestvideo', 
           '-o', '"{}/%(id)s.video.f%(format_id)s.%(ext)s"'.format(video_dir),
           '"{}"'.format(url)]
    os.system(' '.join(cmd))


def download_audio(url, audio_dir):
    cmd = ['youtube-dl', '--ignore-errors', 
           '--download-archive', 'scraping/downloaded_audio.txt', 
           '--format', 'bestaudio', 
           '-o', '"{}/%(id)s.audio.f%(format_id)s.%(ext)s"'.format(audio_dir),
           '"{}"'.format(url)]
    os.system(' '.join(cmd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_list', default='meta/spatialaudiogen_db.lst', help='File containing list of youtube ids.')
    parser.add_argument('--output_dir', default='data/orig', help='Output folder.')
    args = parser.parse_args(sys.argv[1:])

    youtube_ids = open(args.db_list).read().splitlines()
    # select_video_formats(youtube_ids)
    video_fmt = {l.split()[0]: l.strip().split()[1] for l in list(open('scraping/download_formats.txt'))}

    for ii, yid in enumerate(youtube_ids[:400]):
        if yid not in video_fmt:
            continue
        url = 'https://www.youtube.com/watch?v=' + yid
        print '{}/{}: {}'.format(ii, len(youtube_ids), url)

        # download_video(url, args.output_dir)
        download_video_with_fmt(url, video_fmt[yid], args.output_dir)
        download_audio(url, args.output_dir)
        time.sleep(1)
