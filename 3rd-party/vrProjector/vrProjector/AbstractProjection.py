# Copyright 2016 Bhautik J Joshi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# imports
from PIL import Image
import math
import abc
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

class AbstractProjection:
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self.use_bilinear = False
    pass

  def set_use_bilinear(self, val):
    self.use_bilinear = val

  def get_pixel_from_uv(self, u, v, image):
    x = int(self.imsize[0]*u)
    y = int(self.imsize[1]*v)
    x = min(x,self.imsize[0]-1)
    y = min(y,self.imsize[1]-1)
    pix = image[y,x]
    return pix

  @staticmethod
  def _loadImage(imageFile):
    img = Image.open(imageFile)
    imsize = img.size
    npimage = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    return npimage, imsize

  def loadImage(self, imageFile):
    self.image, self.imsize = self._loadImage(imageFile)
    self.set_angular_resolution()

  @staticmethod
  def _initImage(width, height, dtype=np.uint8):
    image = np.ndarray((height, width, 3), dtype=dtype)
    return image

  def initImage(self, width, height, dtype=np.uint8):
    self.image = self._initImage(width, height, dtype=dtype)
    self.imsize = (width, height)
    self.set_angular_resolution()

  @staticmethod
  def _saveImage(img, imgsize, destFile):
    mode = 'RGBA'
    arr = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    img = Image.frombuffer(mode, imgsize, arr.tostring(), 'raw', mode, 0, 1)
    img.save(destFile)

  def saveImage(self, destFile):
    self._saveImage(self.image, self.imsize, destFile)

  # this isn't any faster because of the GIL on the image object
  def reprojectToThisThreaded(self, sourceProjection, numThreads):
    uvList = []
    fx = float(self.imsize[0])
    fy = float(self.imsize[1])

    angleList = [self.angular_position((float(i)/fx,float(j)/fy)) for i in range(self.imsize[0]) for j in range(self.imsize[1])]

    poolAngles = ThreadPool(numThreads)
    image = poolAngles.map(sourceProjection.pixel_value, angleList)
    poolAngles.close()
    poolAngles.join()

    idx = 0
    for x in range(self.imsize[0]):
      for y in range(self.imsize[1]):
        pixel = image[idx]
        if pixel is None:
          print x,y
        else:
          self.image[y,x] = pixel
        idx = idx + 1

  def reprojectToThis(self, sourceProjection):
    for x in range(self.imsize[0]):
      for y in range(self.imsize[1]):
        u = float(x)/float(self.imsize[0])
        v = float(y)/float(self.imsize[1])
        theta, phi = self.angular_position((u,v))
        if theta is None or phi is None:
          pixel = (0,0,0)
        else:
          pixel = sourceProjection.pixel_value((theta, phi))
        self.image[y,x] = pixel

  def point_on_sphere(self, theta, phi):
    r = math.cos(phi)
    return (r*math.cos(theta), r*math.sin(theta), math.sin(phi))

  def pixel_value(self, angle):
    if self.use_bilinear:
      return self._pixel_value_bilinear_interpolated(angle)
    else:
      return self._pixel_value(angle)

  @abc.abstractmethod
  def _pixel_value(self, angle):
    return None

  @abc.abstractmethod
  def angular_position(self, texcoord):
    return None

  @abc.abstractmethod
  def set_angular_resolution(self):
    return None

  @staticmethod
  def bilinear_interpolation(x, y, points):
      '''Interpolate (x,y) from values associated with four points.

      The four points are a list of four triplets:  (x, y, value).
      The four points can be in any order.  They should form a rectangle.

          >>> bilinear_interpolation(12, 5.5,
          ...                        [(10, 4, 100),
          ...                         (20, 4, 200),
          ...                         (10, 6, 150),
          ...                         (20, 6, 300)])
          165.0

      '''
      # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

      points = sorted(points)               # order points by x, then by y
      (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

      if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
          raise ValueError('points do not form a rectangle')
      if not x1 <= x <= x2 or not y1 <= y <= y2:
          raise ValueError('(x, y) not within the rectangle')

      return (q11 * (x2 - x) * (y2 - y) +
              q21 * (x - x1) * (y2 - y) +
              q12 * (x2 - x) * (y - y1) +
              q22 * (x - x1) * (y - y1)
             ) / ((x2 - x1) * (y2 - y1) + 0.0)

  def _pixel_value_bilinear_interpolated(self, angle):
    angleeps = self.angular_resolution/8.0
    pixelA = self._pixel_value((angle[0]-angleeps, angle[1]-angleeps))
    pixelB = self._pixel_value((angle[0]-angleeps, angle[1]+angleeps))
    pixelC = self._pixel_value((angle[0]+angleeps, angle[1]-angleeps))
    pixelD = self._pixel_value((angle[0]+angleeps, angle[1]+angleeps))

    pixelR = self.bilinear_interpolation(0,0, [(-1,-1, pixelA[0]), (-1,1, pixelB[0]), (1,-1, pixelC[0]), (1,1, pixelD[0])])
    pixelG = self.bilinear_interpolation(0,0, [(-1,-1, pixelA[1]), (-1,1, pixelB[1]), (1,-1, pixelC[1]), (1,1, pixelD[1])])
    pixelB = self.bilinear_interpolation(0,0, [(-1,-1, pixelA[2]), (-1,1, pixelB[2]), (1,-1, pixelC[2]), (1,1, pixelD[2])])

    return (int(pixelR), int(pixelG), int(pixelB))
