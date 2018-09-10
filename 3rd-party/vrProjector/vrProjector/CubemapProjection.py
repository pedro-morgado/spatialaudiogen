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


from AbstractProjection import AbstractProjection
from PIL import Image
import math

class CubemapProjection(AbstractProjection):
  def __init__(self):
    AbstractProjection.__init__(self)

  def set_angular_resolution(self):
    # imsize on a face: covers 90 degrees
    #     |\
    #  0.5| \
    #     |  \
    #     -----
    #     1/self.imsize[0]
    #  angular res ~= arctan(1/self.imsize[0], 0.5)
    self.angular_resolution = math.atan2(1/self.imsize[0], 0.5)

  def loadImages(self, front, right, back, left, top, bottom):
    self.front, self.imsize = self._loadImage(front)
    self.right, self.imsize = self._loadImage(right)
    self.back, self.imsize = self._loadImage(back)
    self.left, self.imsize = self._loadImage(left)
    self.top, self.imsize = self._loadImage(top)
    self.bottom, self.imsize = self._loadImage(bottom)
    self.set_angular_resolution()

  def setImages(self, front, right, back, left, top, bottom):
    self.front = front
    self.right = right
    self.back = back
    self.left = left
    self.top = top
    self.bottom = bottom
    self.imsize = self.front.shape
    self.set_angular_resolution()

  def initImages(self, width, height):
    self.imsize = (width, height)
    self.front = self._initImage(width, height)
    self.right = self._initImage(width, height)
    self.back = self._initImage(width, height)
    self.left = self._initImage(width, height)
    self.top = self._initImage(width, height)
    self.bottom = self._initImage(width, height)
    self.set_angular_resolution()

  def saveImages(self, front, right, back, left, top, bottom):
    self._saveImage(self.front, self.imsize, front)
    self._saveImage(self.right, self.imsize, right)
    self._saveImage(self.back, self.imsize, back)
    self._saveImage(self.left, self.imsize, left)
    self._saveImage(self.top, self.imsize, top)
    self._saveImage(self.bottom, self.imsize, bottom)

  def _pixel_value(self, angle):
    theta = angle[0]
    phi = angle[1]
    if theta is None or phi is None:
      return (0,0,0)

    sphere_pnt = self.point_on_sphere(theta, phi)
    x = sphere_pnt[0]
    y = sphere_pnt[1]
    z = sphere_pnt[2]

    eps = 1e-6

    if math.fabs(x)>eps:
      if x>0:
        t = 0.5/x
        u = 0.5+t*y
        v = 0.5+t*z
        if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
          return self.get_pixel_from_uv(u, v, self.front)
      elif x<0:
        t = 0.5/-x
        u = 0.5+t*-y
        v = 0.5+t*z
        if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
          return self.get_pixel_from_uv(u, v, self.back)

    if math.fabs(y)>eps:
      if y>0:
        t = 0.5/y
        u = 0.5+t*-x
        v = 0.5+t*z
        if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
          return self.get_pixel_from_uv(u, v, self.right)
      elif y<0:
        t = 0.5/-y
        u = 0.5+t*x
        v = 0.5+t*z
        if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
          return self.get_pixel_from_uv(u, v, self.left)

    if math.fabs(z)>eps:
      if z>0:
        t = 0.5/z
        u = 0.5+t*y
        v = 0.5+t*-x
        if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
          return self.get_pixel_from_uv(u, v, self.bottom)
      elif z<0:
        t = 0.5/-z
        u = 0.5+t*y
        v = 0.5+t*x
        if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
          return self.get_pixel_from_uv(u, v, self.top)

    return None

  def get_theta_phi(self, _x, _y, _z):
    dv = math.sqrt(_x*_x + _y*_y + _z*_z)
    x = _x/dv
    y = _y/dv
    z = _z/dv
    theta = math.atan2(y, x)
    phi = math.asin(z)
    return theta, phi

  @staticmethod
  def angular_position(texcoord):
    u = texcoord[0]
    v = texcoord[1]
    return None

  def reprojectToThis(self, sourceProjection):
    halfcubeedge = 1.0

    for x in range(self.imsize[0]):
      for y in range(self.imsize[1]):
        u = 2.0*(float(x)/float(self.imsize[0])-0.5)
        v = 2.0*(float(y)/float(self.imsize[1])-0.5)

        # front
        theta, phi = self.get_theta_phi(halfcubeedge, u, v)
        pixel = sourceProjection.pixel_value((theta, phi))
        self.front[y,x] = pixel

        # right
        theta, phi = self.get_theta_phi(-u, halfcubeedge, v)
        pixel = sourceProjection.pixel_value((theta, phi))
        self.right[y,x] = pixel

        # left
        theta, phi = self.get_theta_phi(u, -halfcubeedge, v)
        pixel = sourceProjection.pixel_value((theta, phi))
        self.left[y,x] = pixel

        # back
        theta, phi = self.get_theta_phi(-halfcubeedge, -u, v)
        pixel = sourceProjection.pixel_value((theta, phi))
        self.back[y,x] = pixel

        # bottom
        theta, phi = self.get_theta_phi(-v, u, halfcubeedge)
        pixel = sourceProjection.pixel_value((theta, phi))
        self.bottom[y,x] = pixel

        # top
        theta, phi = self.get_theta_phi(v, u, -halfcubeedge)
        pixel = sourceProjection.pixel_value((theta, phi))
        self.top[y,x] = pixel
