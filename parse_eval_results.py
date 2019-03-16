import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('eval_detailed_fn', help='Output of eval.py script (eval-detailed.log stored inside model directory).')
args = parser.parse_args(sys.argv[1:])

def parse_eval_detailed_file(fn):
    data = open(fn).read().splitlines()
    metrics = data[0].split(' | ')[1].split()
    yids = sorted(list(set([dt.split()[0] for dt in data[1:]])))
    sample_time = {y: [] for y in yids}
    sample_vals = {y: [] for y in yids}
    for dt in data[1:]:
        y, t = dt.split(' | ')[0].split()
        sample_time[y].append(float(t))
        vals = [float(v) for v in dt.split(' | ')[1].split()]
        sample_vals[y].append(vals)
        
    for y in yids:
        sample_time[y] = np.array(sample_time[y])
        sample_vals[y] = np.array(sample_vals[y])

        order = np.argsort(sample_time[y])
        sample_time[y] = sample_time[y][order]
        sample_vals[y] = sample_vals[y][order]

    return sample_vals, sample_time, metrics

detailed_vals, _, keys = parse_eval_detailed_file(args.eval_detailed_fn)
video_ids = detailed_vals.keys()

# Compute video averages
METRICS = ['mse/avg', 'stft/avg', 'env_mse/avg', 'emd/dir']
METRICS_STR = ['MSE ', 'STFT', 'ENV ', 'EMD ']
SAMPLES_PER_SEC = 4800
metrics_foa = {}
for mt in METRICS:
    idx = keys.index(mt)
    if mt in ['emd/dir', 'env_mse/avg', ]:
        vals = [np.sqrt(detailed_vals[vid][:, idx]**2 * SAMPLES_PER_SEC).mean() for vid in video_ids]
    elif mt == 'mse/avg':
        vals = [np.sqrt(detailed_vals[vid][:, idx] * SAMPLES_PER_SEC).mean() for vid in video_ids]
    else:
        vals = [detailed_vals[vid][:, idx].mean() for vid in video_ids]
    metrics_foa[mt] = vals


for mt_str, mt in zip(METRICS_STR, METRICS):
    print("{} = {:.3f}".format(mt_str, np.mean(metrics_foa[mt])))
