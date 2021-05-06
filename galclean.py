# Licensed under GNU General Public License - see LICENSE.rst
'''

    GalClean (https://github.com/astroferreira/galclean)

    This simple tool was developed to remove bright sources
    other than the central galaxy in EFIGI [1] stamps. It uses
    Astropy's Photutils detect_sources alongside some transformations
    with binary masks. It can be used as a standalone
    tool or as a module through the galclean function (see .ipynb
    for an example). The first version of galclean was used in
    10.1093/mnras/stx2266 [2].

    usage:
        python galclean.py <file_path_to_fits>
                [--siglevel SIGLEVEL] [--min_size MIN_SIZE]

        siglevel : float
            float with the number of std deviations
            above the sky background in which to use in the detection step.

        min_size: float
            minimum size in fraction of the original img size to detect
            external sources. If min_size = 0.01, this means 1% of
            the image size. For a image of NxN size this would mean
            (N*0.01)^2 pixels of the upscaled image.

    output:
        segmented fits: <original_name>_seg.fits
        plot with segmentation and residuals: segmentation.png

    Author: Leonardo de Albernaz Ferreira
            leonardo.ferreira@nottingham.ac.uk
            (https://github.com/astroferreira)

    References:
        [1] https://www.astromatic.net/projects/efigi
        [2] http://adsabs.harvard.edu/abs/2018MNRAS.473.2701D

'''
import sys
import argparse
import warnings

import numpy as np
import numpy.ma as ma

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plt

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.stats import biweight_midvariance, mad_std, sigma_clipped_stats
from astropy.utils.exceptions import AstropyDeprecationWarning
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

        Parameters
        ----------
        data : array_like
            2D array of the image.
        iterations : int
            Number of iteractions until returning the
            measurement of the sky background. This is
            a recursive function. When ``iterations`` = 0
            it will return the mean, median, and std.
        mask :  array_like
            2D segmentation array of the sources found
            in previous interaction.
        Returns
        -------
        (mean, median, std) : tuple of floats
            The mean, median and standard deviation of
            the sky background measured in ``data``.
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

        circ_mask = generate_circular_kernel(5)
        next_mask = binary_dilation(segm_img, circ_mask)
        return measure_background(data, iterations-1, next_mask)


def generate_circular_kernel(d):
    '''
        Generates a circular kernel to be used with
        dilation transforms used in GalClean. This
        is done with an 2D matrix of 1s, flipping to
        0 when x^2 + y^2 > r^2.

        Parameters
        ----------
        d : int
            The diameter of the kernel. This should be an odd
            number, but the function handles it in either case.

        Returns
        -------
        circular_kernel : 2D numpy.array bool
            A 2D boolean array with circular shape.
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


def segmentation_map(data, threshold, min_size=0.01):
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

        Parameters
        ----------
        data :  array_like
            The 2D array of the image.
        threshold : float
            Threshold above the sky background in which to
            use as the detection level in detect_sources.
        min_size : float
            Minimum fraction size as a fraction of the image size
            to convert to number of pixels in which is used
            to detect sources. If the source is below
            this size it will not be detected.
        Returns
        -------
        seg_map : array_like
            2D segmentation map of the external sources
            excluding the galaxy in the centre.
    '''

    zp = int(np.round(data.shape[0]/2))

    npixels = int((data.shape[0]*min_size)**2)
    seg_map = detect_sources(data, threshold, npixels=npixels).data

    gal_mask = np.zeros_like(seg_map)

    gal_mask[np.where(seg_map == seg_map[zp, zp])] = 1

    # binary dilation with gal_mask, to make galmask bigger
    gal_mask = binary_dilation(gal_mask, generate_circular_kernel(zp/10))

    background_pixels = data[seg_map == 0]

    seg_map[seg_map == seg_map[zp, zp]] = 0
    seg_map[seg_map > 0] = 1

    seg_map = seg_map - gal_mask

    seg_map[seg_map < 0] = 0
    seg_map[seg_map > 0] = 1

    # binary dilation for sources segmentation map,
    #  zp/20 ~ 2.5% of the image galaxy size
    seg_map = binary_dilation(seg_map, generate_circular_kernel(zp/20))

    return seg_map, background_pixels


def rescale(data, scale_factor):
    '''
        This is simply a wrapper to the zoom
        function of scipy in order to avoid
        oversampling.

        Parameters
        ----------
        data : array_like
            2D array of the image
        scale_factor : float
            Scale factor to apply in the zoom.
        Returns
        -------
        rescaled_data : array_like
            scaled version of ``data`` in which its
            size was changed by ``scale_factor``.
    '''

    if(data.shape[0]*scale_factor > 2000):
        scale_factor = 2000/data.shape[0]

    return zoom(data, scale_factor, prefilter=True)


def galclean(ori_img, std_level=4, min_size=0.01, show=False, save=True):
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

        Parameters
        ----------
        ori_img : array_like
            2D array of the image
        std_level : float
            Number of standard deviations above the sky median
            used to define the threshold for detection.
        min_size : float
            Minimum fraction size as a fraction of the image size
            to convert to number of pixels in which is used
            to detect sources. If the source is below
            this size it will not be detected.
        show: bool
            Used as an option to show or not the inspection
            PNG after segmentation.
        Returns
        -------
        segmented_img : array_like
            2D array of the image after segmentation.

    '''
    mean, median, std = measure_background(ori_img, 2, np.zeros_like(ori_img))
    threshold = median + (std_level*std)

    # upscale the image. It is easier to segment larger sources.
    scaled_img = rescale(ori_img, 4)

    seg_map, background_pixels = segmentation_map(scaled_img, threshold, min_size=min_size)

    # apply segmentation map to the image. Replace segmented regions
    # with sky median
    segmented_img = np.zeros_like(scaled_img)
    segmented_img[seg_map == 0] = scaled_img[seg_map == 0]
    
    n_pix_to_replace = segmented_img[seg_map == 1].shape[0]
    segmented_img[seg_map == 1] = np.random.choice(background_pixels,
                                                   n_pix_to_replace)

    downscale_factor = ori_img.shape[0]/segmented_img.shape[0]
    segmented_img = rescale(segmented_img, downscale_factor)

    plot_result(ori_img, segmented_img, show=show, save=save)
    
    return segmented_img


def galshow(data, ax=None, vmax=99.5, vmin=None):
    '''
        Wrapper to the imshow function of matplotlib
        pyplot. This is just to apply the look and feel
        I generally use with galaxy images representations.

        Parameters
        ----------
        data : array_like
            2D array of the image
        ax : Axis
            Axis in which to plot the image. If ``None``, it
            creates a new Axis.
        vmax: float
            Top fraction in which to use clip the image when plotting
            it. It is used in numpy.percentile.
        vmin: flot
            Bottom fraction in which to clip the image when plotting
            it. It is used in numpy.percentile.
        Returns
        -------
        ax : AxisImage
            The Axis provided as argument after plot or the one
            created for it.
    '''

    if(vmax is not None):
        vmax = np.percentile(data, vmax)

    if(vmin is not None):
        vmin = np.percentile(data, vmin)

    if(ax is None):
        f, ax = plt.subplots(1, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax.imshow(data, vmax=vmax, vmin=vmin, cmap='gist_gray_r',
                     origin='lower')


def plot_result(ori_img, segmented_img, show=False, save=False):
    '''
        Plot the original image, segmented and
        the residual.

        Parameters
        ----------
        ori_img : array_like
            2D array for the original image (before)
            segmentation.
        segmented_img : array_like
            2D array for the segmented image.
        save : bool
            Option used to determine to save the
            ouput image or not.

    '''
    residual = ori_img-segmented_img
    
    if(show):
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        axs[0].set_title('Original Image')
        galshow(ori_img, axs[0])

        axs[1].set_title('Segmented Image')
        galshow(segmented_img, axs[1])

        axs[2].set_title('Original - Segmented')
        galshow(residual, axs[2])

        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()

        if(save):
            fig.savefig('segmentation.png')
            print('Output Inspection PNG {}'.format('segmentation.png'))
            
    if(save):
        np.save('segmentation_map', residual)
        print('Output Segmap Mask {}'.format('segmentation_map.npy'))
        


def __handle_input(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path',
                        help='Path to the galaxy image fits file'
                        )
    parser.add_argument('--siglevel',
                        help='STD above skybackground level \
                              to detect sources. Default: 4',
                        type=float,
                        default=4)
    parser.add_argument('--min_size',
                        help='Minimum size of the sources to be \
                              extracted in fraction of the \
                              image total size (e.g. 0.01 for 1 \
                              per cent of the image size). Default: 0.01',
                        type=float,
                        default=0.01)
    
    parser.add_argument('--show', nargs='?', default=False, const=True)
    parser.add_argument('--save', nargs='?', default=False, const=True)

    args = parser.parse_args()

    print('Running GalClean with siglevel {} and min_size {}'
          .format(args.siglevel, args.min_size))

    return args

if __name__ == '__main__':

    args = __handle_input(sys.argv)

    try:
        ori_img = fits.getdata(args.file_path)

        clean_image = galclean(ori_img, args.siglevel,
                               args.min_size, show=args.show,
                               save=args.save)

        output_path = '{}_seg.fits'. \
                      format(args.file_path.split('.fits')[0])

        fits.writeto(output_path, clean_image, overwrite=True)
        print('Run complete.')
        print('Output fits {}'.format(output_path))
    except FileNotFoundError:
        print('Fits not found with path {}.\
               Please try again with a different path'.format(args.file_path))
