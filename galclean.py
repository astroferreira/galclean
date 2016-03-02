import pyfits as pf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sys
import numpy.ma as ma
import matplotlib.pylab as plt
from astropy.convolution import Gaussian2DKernel
from photutils import detect_sources
from photutils import detect_threshold
from astropy.stats import biweight_midvariance, mad_std
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel
from photutils.detection import detect_sources
from scipy.ndimage import binary_dilation



def measureBackground(data, iterations, mask):
    if(mask.sum() > 0):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
    if(iterations == 0):
        return mean, median, std
    else:
        threshold = median + (std * 2)
        segm_img = detect_sources(data, threshold, npixels=5)
        mask = segm_img.astype(np.bool)    # turn segm_img into a mask
        circMask = generateCircularMask(5)
        finalMask = binary_dilation(mask, circMask)
        return measureBackground(data, iterations-1, finalMask)

def generateCircularMask(d):
    if (d % 2) == 0:
        d = d + 1
    
    mask = np.ones((d, d)) #make a 1s box
    r = np.round(d/2)
    x0 = r
    y0 = r
    for i in range(0, d, 1):
        for j in range(0, d, 1):
            if abs((i-x0))+abs((j-y0)) > r:
                mask[i][j] = 0


    return mask


if len(sys.argv) < 1:
  print '\n\n'
  print sys.argv[0] , 'galaxyFilePath\n'
  exit()
else:
  galaxyFilePath = sys.argv[1]


oriImg = pf.getdata(galaxyFilePath) #loads fits file data
mean, median, std = measureBackground(oriImg, 2, np.zeros_like(oriImg)) #measures background distribution
threshold = median + (3*std) #threshold to detect_sources
segMap = detect_sources(oriImg, threshold, npixels=100) #generate a segmentation map for any source above threshold with number of pixels above npixels
galMask = np.zeros_like(segMap) 
zp = np.round(oriImg.shape[0]/2) #Only works for image with galaxy in the center 
galMask[segMap == segMap[zp, zp]] = 1 #selects the central segmentation to remove from initial segmentation map
finalMask = binary_dilation(galMask, generateCircularMask(zp/10)) #binary convolution with circular mask to get galaxy exterior region zp/10 ~ 5% of the galaxy image size
   


segMap[segMap == segMap[zp, zp]] = 0 #remove galaxy segmentation from segmentation map
segMap[segMap > 0] = 1  #transform segmentation map on binary mask
segMap = segMap - finalMask

#force binary mask after subtraction
segMap[segMap < 0] = 0 
segMap[segMap > 0] = 1 


segMap = binary_dilation(segMap, generateCircularMask(zp/20)) #binary convolution for sources segmentation map zp/20 ~ 2.5% of the image galaxy size
segImg = np.zeros_like(oriImg) + median #add background median to segmented regions
segImg[segMap == 0] = oriImg[segMap == 0] #transfer non-segmented regions to output image

pf.writeto(galaxyFilePath.split('.fits')[0] + '_seg.fits', segImg, clobber=1)
