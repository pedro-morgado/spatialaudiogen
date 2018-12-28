# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read and preprocess image datalib.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image datalib for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
IMAGENET_MEAN = np.array([123., 117., 104.])


def distort_color(image, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  with tf.op_scope([image], scope, 'distort_color'):
    color_ordering = tf.random_uniform((), 0, 2, dtype=tf.int32)

    def distort1(x):
      x = tf.image.random_brightness(x, max_delta=32. / 255.)
      x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
      x = tf.image.random_hue(x, max_delta=0.2)
      return tf.image.random_contrast(x, lower=0.5, upper=1.5)

    def distort2(x):
      x = tf.image.random_brightness(x, max_delta=32. / 255.)
      x = tf.image.random_contrast(x, lower=0.5, upper=1.5)
      x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
      return tf.image.random_hue(x, max_delta=0.2)

    image = tf.cond(tf.equal(color_ordering, 0), lambda : distort1(image), lambda : distort2(image))

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_image(image, height, width, flip_prob=0.5, crop_frac=0.875, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the datalib
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
        image: 3-D float Tensor of image
        height: integer
        width: integer
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor of distorted image used for training.
    """
    with tf.op_scope([image, height, width], scope, 'distort_image'):
        if crop_frac<1.:
            # Random crop
            bbox = tf.reshape(tf.constant([(1-crop_frac)/2., (1-crop_frac)/2., (1+crop_frac)/2., (1+crop_frac)/2.]), (1, 1, 4))
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.86,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.6, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

            # Crop the image to the specified bounding box and resize.
            distorted_image = tf.slice(image, bbox_begin, bbox_size)
        else:
            distorted_image = image

        # Resize to specified size
        distorted_image = tf.image.resize_images(distorted_image, [height, width])

        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        # distorted_image = distort_color(distorted_image/255.) * 255.

        return distorted_image - IMAGENET_MEAN.reshape((1, 1, 3))


def eval_image(image, height, width, center_only=True, flip=False, crop_fraction=1.0, pad2square=False, scope=None):
    """Prepare one image for evaluation.

    Args:
        image: 3-D float Tensor. Single image.
        height: integer. Desired height.
        width: integer. Desired width.
        crop_fraction: float in (0,1]
        center_only: boolean, if True (default) returns only center crop, if False returns center and corner crops.
        flip: boolean, if True returns flipped versions of each crop as well. Default: False.
        scope: Optional scope for op_scope.
    Returns:
        4-D float Tensor of prepared images.
    """
    with tf.op_scope([image, height, width], scope, 'eval_image'):
        assert image.dtype == tf.uint8, ValueError('image should be uint8.')
        assert 0. < crop_fraction <= 1., ValueError('crop fraction should be in (0, 1].')
        if not center_only: assert crop_fraction < 1., ValueError('Corner crops are equal to center crop because crop fraction is 1.')

        if pad2square:
            org_dims = tf.shape(image)
            image = tf.image.resize_image_with_crop_or_pad(image, tf.reduce_max(org_dims[:2]), tf.reduce_max(org_dims[:2]))
        img_dims = tf.shape(image)

        crops = [tf.image.central_crop(image, central_fraction=crop_fraction) if crop_fraction < 1. else image]
        if not center_only:
            crop_dims = tf.to_int32(tf.to_float(img_dims) * crop_fraction)
            bboxes = [[0, 0], [0, img_dims[1]-crop_dims[1]], [img_dims[0]-crop_dims[0], 0], [img_dims[0]-crop_dims[0], img_dims[1]-crop_dims[1]]]
            crops += [tf.image.crop_to_bounding_box(image, bb[0], bb[1], crop_dims[0], crop_dims[1]) for bb in bboxes]
        if flip:
            crops += [tf.image.flip_left_right(c) for c in crops]
        crops = tf.concat(0, [tf.expand_dims(c, 0) for c in crops]) if len(crops)>1 else tf.expand_dims(crops[0], 0)

        crops = tf.image.resize_bilinear(crops, [height, width], align_corners=False) - IMAGENET_MEAN.reshape((1, 1, 1, 3))
        if center_only and not flip:
            crops = tf.squeeze(crops, squeeze_dims=0)

        return crops


def eval_image_py(image, height, width, center_only=True, flip=False, crop_fraction=1.0, pad2square=False, scope=None):
    """Prepare one image for evaluation.

    Args:
        image: 3-D float Tensor. Single image.
        height: integer. Desired height.
        width: integer. Desired width.
        crop_fraction: float in (0,1]
        center_only: boolean, if True (default) returns only center crop, if False returns center and corner crops.
        flip: boolean, if True returns flipped versions of each crop as well. Default: False.
        scope: Optional scope for op_scope.
    Returns:
        4-D float Tensor of prepared images.
    """
    assert image.dtype == np.uint8, ValueError('image should be uint8.')
    assert 0. < crop_fraction <= 1., ValueError('crop fraction should be in (0, 1].')
    if not center_only: assert crop_fraction < 1., ValueError('Corner crops are equal to center crop because crop fraction is 1.')

    if image.ndim == 2:
        image = np.expand_dims(image, 2)
        image = np.concatenate((image, image, image), 2)


    if pad2square:
        org_dims = image.shape
        max_dim = org_dims[:2].max()
        image_new = np.zeros((max_dim, max_dim, 3))
        image_new[(max_dim-org_dims[0])/2:(max_dim-org_dims[0])/2+org_dims[0], (max_dim-org_dims[1])/2:(max_dim-org_dims[1])/2+org_dims[1], :] = image
        image = image_new
    img_dims = [int(s) for s in image.shape]

    if crop_fraction < 1.:
        shape = [max((int(d*crop_fraction), 1)) for d in img_dims[:2]]
        bbox = [int((img_dims[0]-shape[0])/2), int((img_dims[1]-shape[1])/2)]

        crops = [image[bbox[0]:bbox[0]+shape[0], bbox[1]:bbox[1]+shape[1], :]]
    else:
        crops = [image]

    if not center_only:
        crop_dims = [int(d*crop_fraction) for d in img_dims]
        bboxes = [[0, 0], [0, img_dims[1]-crop_dims[1]], [img_dims[0]-crop_dims[0], 0], [img_dims[0]-crop_dims[0], img_dims[1]-crop_dims[1]]]
        crops += [image[bb[0]:bb[0]+shape[0], bb[1]:bb[1]+shape[1], :] for bb in bboxes]
    if flip:
        crops += [np.flip(c, 0) for c in crops]

    import skimage.transform
    crops = [skimage.transform.resize(c, (height, width))*255. - IMAGENET_MEAN.reshape((1, 1, 3)) for c in crops]

    crops = np.concatenate([np.expand_dims(c, 0) for c in crops], 0) #if len(crops)>1 else np.expand_dims(crops[0], 0)
    if center_only and not flip:
        crops = np.squeeze(crops, axis=0)
    return crops
