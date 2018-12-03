## Licensed under GNU General Public License - see LICENSE.rst

'''

    GalClean

    This simple tool was developed to remove bright sources
    other than the central galaxy in EFIGI [1] stamps. It uses
    Astropy's Photutils detect_sources alongside some transformations
    with binary masks/convolution. It can be used as a standalone
    tool or as a module through the galclean function. This was
    published alongside 10.1093/mnras/stx2266 [2].

    The default values are the ones that were used in the paper, I
    refactored it so it would be possible to tweak it for your personal
    goal with different data.

    usage: galclean.py <fits_path> [std above skybg] [min size in % for extraction]

        std above skybg: float with the STD above the sky background in
            which to use in the detection step
        
        min size in % for extractions: minimum size in % of the original img
            for the extracted sources. Images are upscaled before extraction.

    output:
        segmented fits: <original_name>_seg.fits
        plot with segmentation and residuals: segmentation.png

    Author: Leonardo de Albernaz Ferreira leonardo.ferreira@nottingham.ac.uk (https://github.com/astroferreira)
    References:
        [1] https://www.astromatic.net/projects/efigi
        [2] https://ui.adsabs.harvard.edu/#abs/arXiv:1707.02863

'''
import sys

import numpy as np
import numpy.ma as ma

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plt

from astropy.convolution import Gaussian2DKernel
from astropy.stats import biweight_midvariance, mad_std, sigma_clipped_stats
from astropy.io import fits

from photutils import detect_sources, detect_threshold

from scipy.ndimage import binary_dilation, zoom


def measure_background(data, iterations, mask):
    if(mask.sum() > 0):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
    if(iterations == 0):
        return mean, median, std
    else:
        threshold = median + (std * 2)
        segm_img = detect_sources(data, threshold, npixels=5)
        mask = segm_img    # turn segm_img into a mask
        circMask = generate_circular_mask(5)
        finalMask = binary_dilation(mask, circMask)
        return measure_background(data, iterations-1, finalMask)


def generate_circular_mask(d):
    
    d = int(d)

    if (d % 2) == 0:
        d = d + 1

    mask = np.ones((d, d))
    r = np.round(d/2)
    x0 = r
    y0 = r
    for i in range(0, d, 1):
        for j in range(0, d, 1):
            if abs((i-x0))+abs((j-y0)) > r:
                mask[i][j] = 0

    return mask


def segmentation_map(data, threshold, min_size=0.05):

    zp = int(np.round(data.shape[0]/2))

    seg_map = detect_sources(data, threshold, npixels=int(data.shape[0]*min_size)).data

    gal_mask = np.zeros_like(seg_map)
    
    gal_mask[np.where(seg_map == seg_map[zp, zp])] = 1

    final_mask = binary_dilation(gal_mask, generate_circular_mask(zp/10))

    seg_map[seg_map == seg_map[zp, zp]] = 0
    seg_map[seg_map > 0] = 1
    seg_map = seg_map - final_mask

    seg_map[seg_map < 0] = 0
    seg_map[seg_map > 0] = 1

    # binary convolution for sources segmentation map zp/20 ~ 2.5% of the image galaxy size
    seg_map = binary_dilation(seg_map, generate_circular_mask(zp/20))
    
    return seg_map

def galclean(ori_img, std_level, min_size=0.05):

    mean, median, std = measure_background(ori_img, 2, np.zeros_like(ori_img))

    threshold = median + (std_level*std)

    seg_map = segmentation_map(ori_img, threshold, min_size=0.05)

    segmented_img = np.zeros_like(ori_img) + median
    segmented_img[seg_map == 0] = ori_img[seg_map == 0]

    return segmented_img

def plot_result(ori_img, segmented_img):
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    residual = ori_img-segmented_img

    axs[0].set_title('Original Image')
    axs[0].imshow(ori_img, vmax=np.percentile(ori_img, 99))

    axs[1].set_title('Segmented Image')
    axs[1].imshow(segmented_img, vmax=np.percentile(segmented_img, 99))

    axs[2].set_title('Original - Segmented')
    axs[2].imshow(residual, vmax=np.percentile(residual, 99))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=0, wspace=0)

    fig.savefig('segmentation.png')

def rescale(img, scale_factor):

    if(img.shape[0]*scale_factor > 2000):
        scale_factor = 2000/img.shape[0]

    return (zoom(img, scale_factor, prefilter=True), scale_factor)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('usage:', sys.argv[0], 'image_file_path [std above sky level = 3] [min size of source in % of the original img = 5]\n')
        exit()
    else:
        galaxy_file_path = sys.argv[1]
        if(len(sys.argv) > 3):
            std_level = float(sys.argv[2])/100
            min_size = float(sys.argv[3])
        else:
            min_size = 0.05
            std_level = 3

    ori_img = fits.getdata(galaxy_file_path)

    (scaled_img, scale_factor) = rescale(ori_img, 4)

    output_path = '{}_seg.fits'.format(galaxy_file_path.split('.fits')[0])

    (segmented_img, sf) = rescale(galclean(scaled_img, std_level, min_size), 1/scale_factor)

    fits.writeto(output_path, segmented_img, overwrite=True)
    plot_result(ori_img, segmented_img)
