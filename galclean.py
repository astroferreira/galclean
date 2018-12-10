## Licensed under GNU General Public License - see LICENSE.rst

'''

    GalClean

    This simple tool was developed to remove bright sources
    other than the central galaxy in EFIGI [1] stamps. It uses
    Astropy's Photutils detect_sources alongside some transformations
    with binary masks. It can be used as a standalone
    tool or as a module through the galclean function (see .ipynb
    for an example). The first version of galclean was used in
    10.1093/mnras/stx2266 [2].

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
import argparse

import numpy as np
import numpy.ma as ma

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plt

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.stats import biweight_midvariance, mad_std, sigma_clipped_stats

from photutils import detect_sources, detect_threshold

from scipy.ndimage import binary_dilation, zoom


def measure_background(data, iterations, mask):
    '''
        Measure background mean, median and std using an
        recursive/iteractive function. 
        
        This is a little different from examples in 
        Photutils for Background estimation, here I use 
        dilation on sources masks and feed the new mask
        to the next iteration.
    '''

    if(mask.sum() > 0):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
    if(iterations == 0):
        return mean, median, std
    else:
        threshold = median + (std * 2)
        segm_img = detect_sources(data, threshold, npixels=5)
 
        circ_mask = generate_circular_mask(5)
        next_mask = binary_dilation(segm_img, circ_mask)
        return measure_background(data, iterations-1, next_mask)


def generate_circular_mask(d):
    '''
        Generates a circular kernel to be used with
        dilation transforms used in GalClean. This
        is done with an 2D matrix of 1s, flipping to
        0 when x^2 + y^2 > r^2.
    '''
    
    d = int(d)

    if (d % 2) == 0:
        d = d + 1

    mask = np.ones((d, d))
    r = np.round(d/2)
    x0 = r
    y0 = r
    for i in range(0, d, 1):
        for j in range(0, d, 1):
            xx = (i-x0)**2
            yy = (j-y0)**2
            rr = r**2
            if xx + yy > rr:
                mask[i][j] = 0

    return mask


def segmentation_map(data, threshold, min_size=0.05):
    '''
        Generates the segmentation map used to segment
        the galaxy image. This function handles the 
        detection of sources, selection of the galaxy
        mask and the manipulations necessary to remove 
        the galaxy from the segmentation map. Dilation
        transforms are applied to the external sources
        mask and to the galaxy mask. This is done to make
        the segmentation slightly bigger than what is 
        produced by detect_sources. This procedure enable
        galclean to extract the outskirts of sources that
        are bellow the sky background threshold (which is
        generally the case with most sources).
    '''

    zp = int(np.round(data.shape[0]/2))

    seg_map = detect_sources(data, threshold, npixels=int(data.shape[0]*min_size)).data

    gal_mask = np.zeros_like(seg_map)
    
    gal_mask[np.where(seg_map == seg_map[zp, zp])] = 1

    # binary dilation with gal_mask, to make it galmask bigger
    gal_mask = binary_dilation(gal_mask, generate_circular_mask(zp/10))

    seg_map[seg_map == seg_map[zp, zp]] = 0
    seg_map[seg_map > 0] = 1
    seg_map = seg_map - gal_mask

    seg_map[seg_map < 0] = 0
    seg_map[seg_map > 0] = 1

    # binary dilation for sources segmentation map,
    #  zp/20 ~ 2.5% of the image galaxy size
    seg_map = binary_dilation(seg_map, generate_circular_mask(zp/20))
    
    return seg_map

def rescale(img, scale_factor):
    '''
        This is simply a wrapper to the zoom
        function of scipy in order to avoid
        oversampling. 
    '''

    if(img.shape[0]*scale_factor > 2000):
        scale_factor = 2000/img.shape[0]

    return zoom(img, scale_factor, prefilter=True)

def galclean(ori_img, std_level, min_size=0.05, show=False):
    '''
        Galclean measures the sky background, upscales
        the galaxy image, find the segmentation map of 
        sources above the threshold

            threshold = sky median + sky std * level

        where level is usually 3 but can be passed as 
        argument in the function. <min_size> is the 
        minimum size (as a fraction of the image) for 
        the detection of a source. It is converted in the
        segmentation map step to the correspondent number
        of pixels a source must have to be considered for
        detection. It then removes the galaxy from the center,
        apply the segmentation map to the upscaled image,
        replaces external sources with the sky background
        median and then downscale it to the original scale.
    '''
    
    mean, median, std = measure_background(ori_img, 2, np.zeros_like(ori_img))
    threshold = median + (std_level*std)

    #upscale the image. It is easier to segment larger sources.
    scaled_img = rescale(ori_img, 4)

    seg_map = segmentation_map(scaled_img, threshold, min_size=0.05)

    #apply segmentation map to the image. Replace segmented regions
    #with sky median
    segmented_img = np.zeros_like(scaled_img) + median
    segmented_img[seg_map == 0] = scaled_img[seg_map == 0]

    downscale_factor = ori_img.shape[0]/segmented_img.shape[0]
    segmented_img = rescale(segmented_img, downscale_factor)

    if(show):
        plot_result(ori_img, segmentation_map, save=True)
        plt.show()

    return segmented_img

def galshow(img, ax=None, vmax=99.5, vmin=None):
    '''
        Wrapper to the imshow function of matplotlib
        pyplot. This is just to apply the look and feel
        I generally use with galaxy images representations.
    '''

    if(vmax is not None):
        vmax = np.percentile(img, vmax)
        
    if(vmin is not None):
        vmin = np.percentile(img, vmin)

    if(ax is None):
        f, ax = plt.subplots(1, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax.imshow(img, vmax=vmax, vmin=vmin, cmap='gist_gray_r',
                     origin='lower')

def plot_result(ori_img, segmented_img, save=False):
    '''
        Plot the original image, segmented and
        the residual.

    '''
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    residual = ori_img-segmented_img

    axs[0].set_title('Original Image')
    galshow(ori_img, axs[0])

    axs[1].set_title('Segmented Image')
    galshow(segmented_img, axs[1])

    axs[2].set_title('Original - Segmented')
    galshow(residual, axs[2])

    plt.subplots_adjust(hspace=0, wspace=0)
    
    if(save):
        fig.savefig('segmentation.png')

def __handle_input(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', 
                        help='Path to the galaxy image fits file'
                        )
    parser.add_argument('--siglevel', 
                        help='STD above skybackground level \
                              to detect sources. Default: 3', 
                        type=float,
                        default=3)
    parser.add_argument('--min_size', 
                        help='Minimum size of the sources to be \
                              extracted in fraction of the \
                              image total size (e.g. 0.05 for 5 \
                              per cent of the image size). Default: 0.05',
                        type=float,
                        default=0.05)

    args = parser.parse_args()

    print('Running GalClean with siglevel {} and min_size {}'
          .format(args.siglevel, args.min_size))

    return args

if __name__ == '__main__':

    args = __handle_input(sys.argv)

    try:
        ori_img = fits.getdata(args.file_path)

        clean_image = galclean(ori_img, args.siglevel, 
                               args.min_size)

        output_path = '{}_seg.fits'. \
                      format(args.file_path.split('.fits')[0])

        fits.writeto(output_path, clean_image, overwrite=True)
        print('Run complete.')
        print('Output fits {}'.format(output_path))
        print('Output Inspection PNG {}'.format('segmentation.png'))
    except FileNotFoundError:
        print('Fits not found with path {}.\
               Please try again with a different path'.format(args.file_path))