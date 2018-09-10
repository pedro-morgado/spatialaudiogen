# vrProjector, by Bhautik Joshi

vrProjector is a python library and command-line tool to convert from one type of spherical projection to another. Currently it supports converting between Equirectangular, Cubemaps and Side-by-Side fisheye projections.

### Prerequisites

You'll need the python imaging library; I've found it's best to get it via Pillow:

```sh
$ pip install Pillow
```

You'll also need numpy:

```sh
$ pip install numpy
```

### Running vrProjector on the command line

You can run vrProjector simply by running the ```vrProjectorCmd``` shell script on the command-line. You'll need to specify the source and output projection types as well as the source/output images and paramaters themselves. For example, use this to turn an equirectangular image into a set of 128x128 pixel cubemap faces:

```sh
$ ./vrProjectorCmd --sourceProjection Equirectangular --sourceImage images/equirectangular.png --sourceProjection Equirectangular --outProjection CubeMap --outImage "front.png right.png back.png left.png top.png bottom.png" --outWidth 128 --outHeight 128
```

This converts an input equirectangular image into a side-by-side fisheye projection:

```sh
$ ./vrProjectorCmd --sourceProjection Equirectangular --sourceImage images/equirectangular.png --sourceProjection Equirectangular --outProjection SideBySideFisheye --outImage foo.png --outWidth 256 --outHeight 128
```

You can access the full set of available commands via the ```-h``` switch:

```sh
$ ./vrProjectorCmd -h
usage: vrProjectorWrapper.py [-h] --sourceProjection SOURCEPROJECTION
                             --sourceImage SOURCEIMAGE
                             [--useBilnear USEBILNEAR] --outProjection
                             OUTPROJECTION --outImage OUTIMAGE --outWidth
                             OUTWIDTH --outHeight OUTHEIGHT

Reproject photospheres

optional arguments:
  -h, --help            show this help message and exit
  --sourceProjection SOURCEPROJECTION
                        Type of source projection. Valid values are:
                        Equirectangular, Cubemap, SideBySideFisheye
  --sourceImage SOURCEIMAGE
                        Source image[s]. List multiple images in double quotes
                        like so "front.png right.png back.png left.png top.png
                        bottom.png"
  --useBilnear USEBILNEAR
                        Use bilinear interpolation when reprojecting. Valid
                        values are true and false.
  --outProjection OUTPROJECTION
                        Type of output projection. Valid values are:
                        Equirectangular, Cubemap, SideBySideFisheye
  --outImage OUTIMAGE   output image[s]. List multiple images in double quotes
                        like so "front.png right.png back.png left.png top.png
                        bottom.png"
  --outWidth OUTWIDTH   output image[s] width in pixels
  --outHeight OUTHEIGHT
                        output image[s] height in pixels
```

### Running vrProjector in python

First thing to do is to import the vrProjector package:

```python
import vrProjector
```

Now load up your source projection - you'd do this for equirectangular:

```python
source = vrProjector.EquirectangularProjection()
source.loadImage("images/equirectangular.png")
```

or this for a set of cubemap images:

```python
source = vrProjector.CubemapProjection()
source.loadImages("front.png", "right.png", "back.png", "left.png", "top.png", "bottom.png")
```

If you want, you can set up the reprojection to bilinearly sample across the surface of the sphere. This improves the quality of low-resolution images a little but leads to a 4x increase in run-time:

```python
source.set_use_bilinear(True)
```

Now create the output projection - in this case side-by-side fisheye - and save the result:

```python
out = vrProjector.SideBySideFisheyeProjection()
out.initImage(2048,1024)
out.reprojectToThis(source)
out.saveImage("sidebysidefisheye.png")
```

Cubemaps are almost the same:

```python
out = vrProjector.CubemapeProjection()
out.initImages(1024,1024)
out.reprojectToThis(source)
out.saveImages("front.png", "right.png", "back.png", "left.png", "top.png", "bottom.png")
```
