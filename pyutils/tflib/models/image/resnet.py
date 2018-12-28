"""Description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, numpy as np

import sys
sys.path.insert(0, '../../../..')

import tensorflow as tf
from pyutils.tflib.models.image import preprocessing
from pyutils.tflib import wrappers as tfw
import string, re

from collections import OrderedDict

__author__ = 'PedroMorgado'
PWD = os.path.dirname(os.path.realpath(__file__))

class ResNet(object):
    def __init__(self, layers):
        self.layers = layers

    def imagenet_preprocess_ops(self, x, img_dim, distort=True, crop10=False):
        if distort:
            return preprocessing.distort_image(x, img_dim, img_dim)
        else:
            return preprocessing.eval_image(x, img_dim, img_dim, not crop10, crop10, 0.875)

    @staticmethod
    def block(x, is_training, b2a, b2b, b2c, b1=None, downsample=False, name=None, reuse=False):
        with tf.variable_scope(name, values=[x, is_training]):
            s = 2 if downsample else 1
            y1 = tfw.conv_2d(x, b1, 1, s, use_bias=False, use_batch_norm=True, activation_fn=None, is_training=is_training, trainable=is_training, reuse=reuse, name='branch1') if b1 is not None else x
            y2 = tfw.conv_2d(x, b2a, 1, s, use_bias=False, use_batch_norm=True, activation_fn=tf.nn.relu, is_training=is_training, trainable=is_training, reuse=reuse, name='branch2a')
            y2 = tfw.conv_2d(y2, b2b, 3, 1, use_bias=False, use_batch_norm=True, activation_fn=tf.nn.relu, is_training=is_training, trainable=is_training, reuse=reuse, name='branch2b')
            y2 = tfw.conv_2d(y2, b2c, 1, 1, use_bias=False, use_batch_norm=True, activation_fn=None, is_training=is_training, trainable=is_training, reuse=reuse, name='branch2c')

            return tf.nn.relu(y1+y2)

    def restore_pretrained(self, inp_shape, scope=None):
        pretrained = np.load(os.path.join(PWD, 'resnet%d.npy' % self.layers)).all()

        w_init, b_init, bn_init = {}, {}, {}
        for k in pretrained.iterkeys():
            if k == 'conv1':
                w_init[k] = pretrained[k]['weights'][:, :, :inp_shape, :]
                bn_init[k] = {'moving_mean': pretrained['bn_'+k]['mean'],
                              'moving_variance': pretrained['bn_'+k]['variance'],
                              'gamma': pretrained['bn_'+k]['scale'],
                              'beta': pretrained['bn_'+k]['offset']}

            elif k == 'fc1000':
                w_init[k] = pretrained[k]['weights']
                b_init[k] = pretrained[k]['biases']

            else:
                l = re.search('res(.*)', k)
                if l is None:
                    continue    # BN params
                l = l.group(1)
                w_init[k] = pretrained[k]['weights']
                bn_init[k] = {'moving_mean': pretrained['bn'+l]['mean'],
                              'moving_variance': pretrained['bn'+l]['variance'],
                              'gamma': pretrained['bn'+l]['scale'],
                              'beta': pretrained['bn'+l]['offset']}

        # print(w_init.keys())
        # print(b_init.keys())
        restore_ops = []
        for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
            if scope is not None and not var.op.name.startswith(scope):
                continue
            name = var.op.name.split('/')
            if name[-1] == 'weights':
                if name[-2] in w_init:
                    shape = var.get_shape().as_list()
                    w = w_init[name[-2]].reshape(shape)
                    restore_ops.append(tf.assign(var, w))
                elif name[-3].startswith('res'):
                    shape = var.get_shape().as_list()
                    w = w_init['_'.join(name[-3:-1])].reshape(shape)
                    restore_ops.append(tf.assign(var, w))

            elif name[-1] == 'biases':
                if name[-2] in b_init:
                    b = b_init[name[-2]]
                    restore_ops.append(tf.assign(var, b))
                elif name[-3].startswith('res'):
                    b = b_init['_'.join(name[-3:-1])]
                    restore_ops.append(tf.assign(var, b))

            elif name[-2] == 'bn':
                if name[-3] in bn_init:
                    bn = bn_init[name[-3]]
                    restore_ops.append(tf.assign(var, bn[name[-1]]))
                elif name[-4].startswith('res'):
                    bn = bn_init['_'.join(name[-4:-2])]
                    restore_ops.append(tf.assign(var, bn[name[-1]]))

            else:
                raise ValueError

        return  restore_ops



class ResNet18(object):
    def imagenet_preprocess_ops(self, x, img_dim, distort=True, crop10=False):
        if distort:
            x = preprocessing.distort_image(x, img_dim, img_dim)
        else:
            x = preprocessing.eval_image(x, img_dim, img_dim, not crop10, crop10, 0.875)

        x = (x + preprocessing.IMAGENET_MEAN.reshape((1, 1, 3)))/255.
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        x = (x - imagenet_mean) / imagenet_std
        return x

    def inference_ops(self, x, is_training=True, spatial_squeeze=True, truncate_at=None, reuse=False):
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        ends = OrderedDict()
        with tf.variable_scope('conv1'):
            name = 'conv'
            ends[name] = x = tfw.conv_2d(x, filters[0], kernels[0], strides[0], use_bias=False, use_batch_norm=True, 
                activation_fn=tf.nn.relu, is_training=is_training, trainable=is_training, reuse=reuse, name=name)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
            if truncate_at is not None and truncate_at == 'conv1':
                return x, ends

        # conv2_x
        name = 'conv2_1'
        ends[name] = x = self._residual_block(x, is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        name = 'conv2_2'
        ends[name] = x = self._residual_block(x, is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        # conv3_x
        name = 'conv3_1'
        ends[name] = x = self._residual_block_first(x, filters[2], strides[2], is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        name = 'conv3_2'
        ends[name] = x = self._residual_block(x, is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        # conv4_x
        name = 'conv4_1'
        ends[name] = x = self._residual_block_first(x, filters[3], strides[3], is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        name = 'conv4_2'
        ends[name] = x = self._residual_block(x, is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        # conv5_x
        name = 'conv5_1'
        ends[name] = x = self._residual_block_first(x, filters[4], strides[4], is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends
            
        name = 'conv5_2'
        ends[name] = x = self._residual_block(x, is_training, reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at is not None and truncate_at == name:
            return x, ends

        # Logit
        with tf.variable_scope('logits') as scope:
            x = tf.reduce_mean(x, [1, 2])
            name = 'fc'
            ends[name] = x = tfw.fully_connected(x, 1000, name=name)
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        return x, ends

    def _residual_block_first(self, x, out_channel, strides, is_training=False, reuse=False, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = tfw.conv_2d(x, out_channel, 1, strides, use_bias=False, 
                    activation_fn=None, is_training=is_training, trainable=is_training, reuse=reuse, name='shortcut')

            # Residual
            x = tfw.conv_2d(x, out_channel, 3, strides, use_bias=False, use_batch_norm=True, 
                activation_fn=tf.nn.relu, is_training=is_training, trainable=is_training, reuse=reuse, name='conv_1')
            x = tfw.conv_2d(x, out_channel, 3, 1, use_bias=False, use_batch_norm=True, 
                activation_fn=None, is_training=is_training, trainable=is_training, reuse=reuse, name='conv_2')

            # Merge
            x = tf.nn.relu(x + shortcut)
        return x

    def _residual_block(self, x, is_training=False, reuse=False, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            shortcut = x
            # Residual
            x = tfw.conv_2d(x, num_channel, 3, 1, use_bias=False, use_batch_norm=True, 
                activation_fn=tf.nn.relu, is_training=is_training, trainable=is_training, reuse=reuse, name='conv_1')
            x = tfw.conv_2d(x, num_channel, 3, 1, use_bias=False, use_batch_norm=True, 
                activation_fn=None, is_training=is_training, trainable=is_training, reuse=reuse, name='conv_2')

            x = tf.nn.relu(x + shortcut)
        return x

    def restore_pretrained(self, inp_shape=3, scope=None):
        pretrained = np.load(os.path.join(PWD, 'resnet18.npy')).all()

        restore_ops = []
        for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
            if scope is not None and not var.op.name.startswith(scope):
                continue

            name = var.op.name[len(scope)+1:] if scope is not None else var.op.name
            restore_ops.append(tf.assign(var, pretrained[name]))

        return  restore_ops

class ResNet50(ResNet):
    def __init__(self):
        ResNet.__init__(self, 50)

    def inference_ops(self, x, is_training=True, for_imagenet=True, spatial_squeeze=True, truncate_at=None, reuse=False):
        inp_dims = x.get_shape()
        assert inp_dims.ndims in (3, 4)
        if inp_dims.ndims == 3: x = tf.expand_dims(x, 0)

        ends = OrderedDict()

        # Scale 1
        name = 'conv1'
        ends[name] = x = tfw.conv_2d(x, 64, 7, 2, use_bias=False, use_batch_norm=True, activation_fn=tf.nn.relu, is_training=is_training, trainable=is_training, reuse=reuse, name=name)
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))
        if truncate_at == name: return x, ends

        name = 'pool1'
        ends[name] = x = tfw.max_pool2d(x, 3, 2, padding='SAME', name=name)
        if truncate_at == name: return x, ends
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        # Scale 2
        for i, c in enumerate(string.ascii_lowercase[:3]):
            name = 'res2'+c
            ends[name] = x = self.block(x, is_training, 64, 64, 256, None if i>0 else 256, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        # Scale 3
        for i, c in enumerate(string.ascii_lowercase[:4]):
            name = 'res3'+c
            ends[name] = x = self.block(x, is_training, 128, 128, 512, 512 if i==0 else None, downsample=True if i==0 else False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        # Scale 4
        for i, c in enumerate(string.ascii_lowercase[:6]):
            name = 'res4'+c
            ends[name] = x = self.block(x, is_training, 256, 256, 1024, 1024 if i==0 else None, downsample=True if i==0 else False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        # Scale 5
        for i, c in enumerate(string.ascii_lowercase[:3]):
            name = 'res5'+c
            ends[name] = x = self.block(x, is_training, 512, 512, 2048, 2048 if i==0 else None, downsample=True if i==0 else False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        name = 'pool5'
        x = tfw.avg_pool2d(x, 7, 1, 'VALID', name=name)
        if spatial_squeeze and x.get_shape().as_list()[1]==x.get_shape().as_list()[2]==1:
            x = tf.squeeze(x, squeeze_dims=[1, 2])
        ends[name] = x
        if truncate_at == name: return x, ends
        print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        # Logits
        if for_imagenet:
            name = 'fc1000'
            ends['logits'] = x = tfw.fully_connected(x, 1000, activation_fn=None, is_training=is_training, trainable=is_training, reuse=reuse, name=name)
            print(' * {:15s} | {:20s} | {:10s}'.format(name, str(x.get_shape()), str(x.dtype)))

        return x, ends


class ResNet101(ResNet):
    def __init__(self):
        ResNet.__init__(self, 101)

    def inference_ops(self, x, is_training=True, for_imagenet=True, spatial_squeeze=True, truncate_at=None, reuse=False):
        inp_dims = x.get_shape()
        assert inp_dims.ndims in (3, 4)
        if inp_dims.ndims == 3: x = tf.expand_dims(x, 0)

        ends = OrderedDict()

        # Scale 1
        name = 'conv1'
        ends[name] = x = tfw.conv_2d(x, 64, 7, 2, use_bias=False, use_batch_norm=True, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, name=name)
        if truncate_at == name: return x, ends

        name = 'pool1'
        ends[name] = x = tfw.max_pool2d(x, 3, 2, padding='SAME', name=name)
        if truncate_at == name: return x, ends

        # Scale 2
        for i, c in enumerate(string.ascii_lowercase[:3]):
            name = 'res2'+c
            ends[name] = x = self.block(x, is_training, 64, 64, 256, None if i>0 else 256, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        # Scale 3
        name = 'res3a'
        ends[name] = x = self.block(x, is_training, 128, 128, 512, 512, downsample=True, reuse=reuse, name=name)
        if truncate_at == name: return x, ends

        for i in range(3):
            name = 'res3b%d' %  (i+1)
            ends[name] = x = self.block(x, is_training, 128, 128, 512, None, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        # Scale 4
        name = 'res4a'
        ends[name] = x = self.block(x, is_training, 256, 256, 1024, 1024, downsample=True, reuse=reuse, name=name)
        if truncate_at == name: return x, ends

        for i in range(22):
            name = 'res4b%d' %  (i+1)
            ends[name] = x = self.block(x, is_training, 256, 256, 1024, None, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        # Scale 5
        for i, c in enumerate(string.ascii_lowercase[:3]):
            name = 'res5'+c
            ends[name] = x = self.block(x, is_training, 512, 512, 2048, 2048 if i==0 else None, downsample=True if i==0 else False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        name = 'pool5'
        x = tfw.avg_pool2d(x, 7, 1, 'VALID', name=name)
        if spatial_squeeze and x.get_shape().as_list()[1]==x.get_shape().as_list()[2]==1:
            x = tf.squeeze(x, squeeze_dims=[1, 2])
        ends[name] = x
        if truncate_at == name: return x, ends

        # Logits
        if for_imagenet:
            name = 'fc1000'
            ends['logits'] = x = tfw.fully_connected(x, 1000, activation_fn=None, is_training=is_training, reuse=reuse, name=name)

        return x, ends


class ResNet152(ResNet):
    def __init__(self):
        ResNet.__init__(self, 152)

    def inference_ops(self, x, is_training=True, for_imagenet=True, spatial_squeeze=True, truncate_at=None, reuse=False):
        inp_dims = x.get_shape()
        assert inp_dims.ndims in (3, 4)
        if inp_dims.ndims == 3: x = tf.expand_dims(x, 0)

        ends = OrderedDict()

        # Scale 1
        name = 'conv1'
        ends[name] = x = tfw.conv_2d(x, 64, 7, 2, use_bias=False, use_batch_norm=True, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, name=name)
        if truncate_at == name: return x, ends

        name = 'pool1'
        ends[name] = x = tfw.max_pool2d(x, 3, 2, padding='SAME', name=name)
        if truncate_at == name: return x, ends

        # Scale 2
        for i, c in enumerate(string.ascii_lowercase[:3]):
            name = 'res2'+c
            ends[name] = x = self.block(x, is_training, 64, 64, 256, None if i>0 else 256, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        # Scale 3
        name = 'res3a'
        ends[name] = x = self.block(x, is_training, 128, 128, 512, 512, downsample=True, reuse=reuse, name=name)
        if truncate_at == name: return x, ends

        for i in range(7):
            name = 'res3b%d' %  (i+1)
            ends[name] = x = self.block(x, is_training, 128, 128, 512, None, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        # Scale 4
        name = 'res4a'
        ends[name] = x = self.block(x, is_training, 256, 256, 1024, 1024, downsample=True, reuse=reuse, name=name)
        if truncate_at == name: return x, ends

        for i in range(35):
            name = 'res4b%d' %  (i+1)
            ends[name] = x = self.block(x, is_training, 256, 256, 1024, None, downsample=False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        # Scale 5
        for i, c in enumerate(string.ascii_lowercase[:3]):
            name = 'res5'+c
            ends[name] = x = self.block(x, is_training, 512, 512, 2048, 2048 if i==0 else None, downsample=True if i==0 else False, reuse=reuse, name=name)
            if truncate_at == name: return x, ends

        name = 'pool5'
        x = tfw.avg_pool2d(x, 7, 1, 'VALID', name=name)
        if spatial_squeeze and x.get_shape().as_list()[1]==x.get_shape().as_list()[2]==1:
            x = tf.squeeze(x, squeeze_dims=[1, 2])
        ends[name] = x
        if truncate_at == name: return x, ends

        # Logits
        if for_imagenet:
            name = 'fc1000'
            ends['logits'] = x = tfw.fully_connected(x, 1000, activation_fn=None, is_training=is_training, reuse=reuse, name=name)

        return x, ends



def main():
    g = tf.Graph()
    with g.as_default():
        model = ResNet18()
        fn_t = tf.constant('test_images/woman.jpg', dtype=tf.string)
        image_t = tf.image.decode_jpeg(tf.read_file(fn_t), channels=3)
        print(image_t)
        image_prep_t = model.imagenet_preprocess_ops(image_t, 224, distort=False)
        image_prep_t = tf.expand_dims(image_prep_t, 0)
        logits_t, ends_t = model.inference_ops(image_prep_t, False)
        # train_vars = tf.global_variables()
        # for v in train_vars:
        #     print(v.op.name, v.get_shape())
        restore_ops = model.restore_pretrained()

    with tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(restore_ops)
        logits_v, img, img_prep = sess.run([logits_t, image_t, image_prep_t])

    class_info = [l.strip() for l in open(os.path.join(PWD, 'imagenet-classes.txt'))]
    top = np.argsort(-logits_v[0])
    top_crop10 = np.argsort(-logits_v.mean(axis=0))
    print(logits_v.shape)
    print(top.shape)

    print('single crop prediction')
    print([class_info[i] for i in top[:5]])

    print('10 crops prediction')
    print([class_info[i] for i in top_crop10[:5]])

    from matplotlib import pyplot as plt
    from pyutils.tflib.models.image.preprocessing import IMAGENET_MEAN
    plt.imshow((img_prep + IMAGENET_MEAN.reshape((1, 1, 3)))/255.)
    # plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
