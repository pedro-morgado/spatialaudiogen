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


import argparse

import vrProjector

def main():
  parser = argparse.ArgumentParser(description='Reproject photospheres')
  parser.add_argument('--sourceProjection', required=True, help='Type of source projection. Valid values are: Equirectangular, Cubemap, SideBySideFisheye')
  parser.add_argument('--sourceImage', required=True, help='Source image[s]. List multiple images in double quotes like so "front.png right.png back.png left.png top.png bottom.png"')
  parser.add_argument('--useBilnear', required=False, help='Use bilinear interpolation when reprojecting. Valid values are true and false.')
  parser.add_argument('--outProjection', required=True, help='Type of output projection. Valid values are: Equirectangular, Cubemap, SideBySideFisheye')
  parser.add_argument('--outImage', required=True, help='output image[s]. List multiple images in double quotes like so "front.png right.png back.png left.png top.png bottom.png"')
  parser.add_argument('--outWidth', required=True, help='output image[s] width in pixels')
  parser.add_argument('--outHeight', required=True, help='output image[s] height in pixels')

  args = parser.parse_args()

  source = None
  if args.sourceProjection.lower() == "Equirectangular".lower():
    source = vrProjector.EquirectangularProjection()
    source.loadImage(args.sourceImage)
  elif args.sourceProjection.lower() == "SideBySideFisheye".lower():
    source = vrProjector.SideBySideFisheyeProjection()
    source.loadImage(args.sourceImage)
  elif args.sourceProjection.lower() == "Cubemap".lower():
    source = vrProjector.CubemapProjection()
    imageList = args.sourceImage.split(' ')
    source.loadImages(imageList[0], imageList[1], imageList[2], imageList[3], imageList[4], imageList[5])
  else:
    print "Quitting because unsupported source projection type: " + args.sourceProjection
    return

  if args.useBilnear is not None:
    if args.useBilnear.lower() == "true":
      source.set_use_bilinear(True)

  out = None
  if args.outProjection.lower() == "Equirectangular".lower():
    out = vrProjector.EquirectangularProjection()
    out.initImage(int(args.outWidth), int(args.outHeight))
  elif args.outProjection.lower() == "SideBySideFisheye".lower():
    out = vrProjector.SideBySideFisheyeProjection()
    out.initImage(int(args.outWidth), int(args.outHeight))
  elif args.outProjection.lower() == "Cubemap".lower():
    out = vrProjector.CubemapProjection()
    out.initImages(int(args.outWidth), int(args.outHeight))
  else:
    print "Quitting because unsupported output projection type: " + args.outProjection
    return

  out.reprojectToThis(source)
  # out.reprojectToThisThreaded(source, 16)

  if args.outProjection.lower() == "Cubemap".lower():
    imageList = args.outImage.split(' ')
    out.saveImages(imageList[0], imageList[1], imageList[2], imageList[3], imageList[4], imageList[5])
  else:
    out.saveImage(args.outImage)

if __name__ == "__main__":
    main()
