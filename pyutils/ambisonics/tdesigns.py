import re
import numpy as np

hpp_file = '../ambisonics/src/tDesigns.hpp'

hpp_corpus = [l.strip() for l in open(hpp_file, 'r')]
tDesign_init, tDesign_end, tDesign_order, tDesign_num_speakers = [], [], [], []
for i, l in enumerate(hpp_corpus):
    m = re.match('array\.clear\(\);', l)
    if m is not None:
        tDesign_init.append(i)

    m = re.match('m_tDesigns\.insert\(std::make_pair\((\d+),\s*array\)\);', l)
    if m is not None:
        tDesign_end.append(i)
        tDesign_order.append(int(m.group(1)))
        tDesign_num_speakers.append(tDesign_end[-1] - tDesign_init[-1] - 1)

tDesigns = []
for init, end, dim in zip(tDesign_init, tDesign_end, tDesign_order):
    tDesign = np.zeros((end-init-1, 3))
    for i, l in enumerate(hpp_corpus[init+1:end]):
        m = re.match('array\.push_back\(Position\((-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*Position::CARTESIAN\)\);', l)
        tDesign[i] = np.array([float(m.group(s)) for s in range(1, 4)])
    tDesigns.append(tDesign)


def get_tDesign(order, num_speakers=None):
    import random
    tDesigns_index = [idx for idx, o in enumerate(tDesign_order) if o == order]
    if num_speakers is not None:
        tDesigns_index = [idx for idx in tDesigns_index if tDesign_num_speakers[idx] == num_speakers]
    if not tDesigns_index:
        raise ValueError('tDesign not available.')
    random.shuffle(tDesigns_index)
    return tDesigns[tDesigns_index[0]]
