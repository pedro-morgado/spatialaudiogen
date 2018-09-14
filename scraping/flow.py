import os
from math import ceil
import numpy as np
import tempfile

# Adapted from run-flownet-many.py from original code https://github.com/lmb-freiburg/flownet2.git
# Needs FlowNet2 to be installed and to be the only caffe installation in the python library path.
class FlowNet2:
    def __init__(self, height, width, caffemodel, deployproto, gpu=0):
        import caffe

        divisor = 64.
        vars = {'TARGET_WIDTH': width,
                'TARGET_HEIGHT': height,
                'ADAPTED_WIDTH': int(ceil(width/divisor) * divisor),
                'ADAPTED_HEIGHT': int(ceil(height/divisor) * divisor)}
        vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
        vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        proto = open(deployproto).readlines()
        for line in proto:
            for key, value in vars.items():
                tag = "$%s$" % key
                line = line.replace(tag, str(value))
            tmp.write(line)
        tmp.flush()

        caffe.set_logging_disabled()
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.net = caffe.Net(tmp.name, caffemodel, caffe.TEST)

    def compute(self, i1, i2):
        def prep_img(img):
            if img.ndim == 2:
                return img[np.newaxis, np.newaxis, :, :]
            else:
                return img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]

        input_dict = {self.net.inputs[0]: prep_img(i1),
                      self.net.inputs[1]: prep_img(i2)}

        # There is some non-deterministic nan-bug in caffe [SOURCE: FLOWNET2]
        for i in range(5):
            self.net.forward(**input_dict)

            # Check for NaNs in results
            has_nans = False
            for name in self.net.blobs:
                if np.isnan(self.net.blobs[name].data[...]).any():
                    has_nans = True
                    break
            if not has_nans:
                break

        flow = self.net.blobs['predict_flow_final'].data
        flow = np.squeeze(flow).transpose((1, 2, 0))
        return flow
