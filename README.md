# GalClean

Simple tool to remove bright sources (excluding the galaxy itself) from a fits image using Astropy PhotUtils.

GalClean can be used as a script following the instructions bellow or as a module (refer to the .ipynb in the repository for an example).


## usage

```shell
python galclean.py <file_path_to_fits> [--siglevel SIGLEVEL] [--min_size MIN_SIZE]
```

## output

1. Segmented input in a fits file: <path>_seg.png
2. PNG for inspection with the original and segmented images with a residual plot

## Reference
This tool was created to remove external bright sources from EFIGI's stamps to avoid source blending
after applying artificial redshift to them with FERENGI. An early version [2] of GalClean was used in [1]. 

[1] [de Albernaz Ferreira & Ferrari, 2018](http://adsabs.harvard.edu/abs/2018MNRAS.473.2701D) \
[2] [First Version](https://github.com/astroferreira/galclean/blob/44ecb2cf4902133c27c4d357b9f72f951b4d5d04/galclean.py)
