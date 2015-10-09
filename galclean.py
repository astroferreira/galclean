import pyfits as pf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sys
import colormaps as cmaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pylab as plt
from astropy.convolution import Gaussian2DKernel
from photutils import detect_sources
from photutils import detect_threshold
import matplotlib.ticker as ticker
from astropy.stats import biweight_midvariance, mad_std
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel
from photutils.detection import detect_sources
from scipy.ndimage import binary_dilation



def printImage(img, ax=False, colorbar=True, vmin=0, vmax=100):
    if(not ax):
        f, ax = plt.subplots(1,1, figsize=(5, 5))
        im = ax.imshow(img, vmin=np.percentile(img,vmin), vmax=np.percentile(img,vmax), interpolation='nearest', cmap=cmaps.viridis)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(img,  vmin=np.percentile(img,vmin), vmax=np.percentile(img,vmax), interpolation='nearest', cmap=cmaps.viridis)

    if(colorbar):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax,  format=ticker.FuncFormatter(fmt))
    return im

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17,
        }


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



outputImg = True

if len(sys.argv) < 2:
  print '\n\n'
  print sys.argv[0] , 'galaxyFilePath outputImg?\n'
  exit()
else:
  galaxyFilePath = sys.argv[1]
  if(sys.argv[2]):
    outputImg    = sys.argv[2].astype(bool)



oriImg = pf.getdata(galaxyFilePath) #loads fits file data
mean, median, std = measureBackground(oriImg, 2, np.zeros_like(oriImg)) #measures background distribution
threshold = median + (3*std) #threshold to detect_sources

segMap = detect_sources(oriImg, threshold, npixels=100) #generate a segmentation map for any source above threshold with number of pixels above npixels
 
galMask = np.zeros_like(segMap) 

zp = np.round(oriImg.shape[0]/2) #denerian operation, works for image with galaxy in the center 

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


#Generates output
if(outputImg):
  f, axes = plt.subplots(1, 4, figsize=(15, 4))

  ax = axes.flat


  printImage(oriImg, ax[0], vmin=0.5, vmax=99.5, colorbar=False)
  printImage(segMap-finalMask*2, ax[1], colorbar=False)
  printImage(segImg, ax[2], vmin=0.5, vmax=99.5, colorbar=False)
  printImage((oriImg-segImg), ax[3], vmin=0.5, vmax=99.5, colorbar=False)


  axes[0].set_ylabel(r'$\rm '+galaxyFilePath+'$', fontdict=font)
  axes[0].set_title(r'$\rm Original $', fontdict=font)
  axes[1].set_title(r'$\rm Segmentation \ Map $', fontdict=font)
  axes[2].set_title(r'$\rm After \ Mask $', fontdict=font)
  axes[3].set_title(r'$\rm Original \ - \ Masked $', fontdict=font)
  plt.tight_layout()
  plt.savefig(galaxyFilePath + '.png', dpi=50)
  plt.close()

pf.writeto(galaxyFilePath.split('.fits')[0] + '_seg.fits', segImg, clobber=1)


