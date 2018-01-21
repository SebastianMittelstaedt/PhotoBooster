# PhotoBooster

With this tool, you can put all your vacation photos in one folder and then automatically optimize them. PhotoBooster performs white balance and saturation boost to remove any issues in illumination in your photo and also makes the colors look more saturated.

PhotoBooster uses the 
* CAT02 color space for chromatic adaption
* HSI color space for saturation boost

It is implemented in OpenCL (using [pyopencl](https://mathema.tician.de/software/pyopencl/)).

# Getting started
* Clone
* Install [pyopencl](https://wiki.tiker.net/PyOpenCL/Installation)
* Install numpy
* Install imageio
* Change last line in PhotoBooster.py to set your folders. bootstrap_all_files_in_folder(ctx, queue, "D:\Test", "D:\Test_output")

It is far away from any usability but for now I don't care. Sorry.

If you are brave enough to use it and / or contribute --> thumbs up :) .

# Examples

![Swan](examples/swan.JPG)
![Swan Boosted](examples/swan_boosted.JPG)
![Landscape](examples/landscape.JPG)
![Landscape Boosted](examples/landscape_boosted.JPG)