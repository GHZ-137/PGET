"""
Obtain Mu (attenuation) mask by blobs detection.

"""
###############################
# Imports
###############################
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.draw import circle_perimeter, circle_perimeter_aa, disk
from skimage.feature import blob_log
from skimage.color import gray2rgb

from skimage.measure import label, regionprops

from functions_images import *
from config import *

###############################
# Variables
###############################
# 95% Treshold      
t_value = 95.  #Intensity tresh. 91.3
#2 -> 93.1
#5-> 86 
min_sigma = .8        #LOG sigmas 
max_sigma = 1.5,
num_sigma = 2
overlap = 0.01      #Overlap fract.
threshold= 0.05     #Detection tresh

# Treshold
img = imread(out_folder + '/corrected.png')
img2 = np.copy(img)

img2 = rescale2(-(-np.exp(img2 / 255.)))

#img2 = rescale(img2, 2., anti_aliasing=True)
tresh = np.percentile(img2, t_value)
rr, cc = np.where(img2 < tresh)
img2[rr,cc] = 0
#rr, cc = np.where(img2 >= tresh)
#img2[rr,cc] = 255

imsave(out_folder + '/tresh.png', img2)
#img2 = imread(out_folder + '/tresh_.png',1)

img3 = np.copy(img2)
rr, cc = np.where(img3 > tresh)
img3[rr,cc] = 255
imsave(out_folder + '/tresh_bin.png', img3)

"""
# Use regions
labels = label(img3) #, connectivity = 1)
vals = np.unique(labels)[1:] # Exclude background
blobs = np.zeros((len(vals), 3))
for cont in range(len(vals)):
    rr, cc = np.where(labels == vals[cont])
    #print (len(rr))

    mean_x = np.mean(rr)
    mean_y = np.mean(cc)
    blobs[cont] = [mean_x, mean_y, r]

"""
# Use blobs
# Blobs as integer coordinates with r = 2
blobs = blob_log(img2, min_sigma = min_sigma,
                 max_sigma = max_sigma,
                 num_sigma = num_sigma,
                 overlap = overlap,
                 threshold= threshold)


blobs[:,2] = r
blobs = blobs[:,:3]
#blobs[:,:2] = 0.5 * blobs[:,:2]
#tresh = rescale(tresh, .5, anti_aliasing=True)

##blobs = adjust_blobs(blobs)


blobs = np.around(blobs, 0).astype(np.int)
print('Blobs:', len(blobs))

# Save blobs values
f = open(out_folder + '/blobs', 'wb')
pickle.dump(blobs, f)
f = open(out_folder + '/blobs', 'rb'); kk = pickle.load(f); f.close()

img3 = gray2rgb(img)
img3 = create_circles(blobs, img3)
mask = create_mask(blobs, [255] * len(blobs), img.shape)


# Crop blobs and mask images
img3, blobs2, borders = crop(img3, blobs, pget_d, r)
mask, _, _ = crop(mask, blobs, pget_d, r)

img3 = resize_max(img3, np.array(img3.shape)*4)
mask = resize_max(mask, np.array(mask.shape)*4)

img3 = add_borders(img3, borders[0], borders[1], 4)
mask = add_borders(mask, borders[0], borders[1], 4)

imsave(out_folder + '/blobs.png', img3)
imsave(out_folder + '/mask.png', mask.astype(np.uint8) )

# Activity image
levels = get_levels(img, blobs)
blobs4 = blobs * 4
act_img = activity_image(blobs4, levels, pget_d, r*4)
act_img = add_borders(act_img, borders[0], borders[1], 4)
act_img = add_scale(act_img, levels / 255., n=4, dec =0)
imsave(out_folder + '/evolution/activity_0.png', act_img)

"""
from skimage.filters import threshold_local, threshold_otsu, rank
from skimage.morphology import disk

radius = 5
footprint = disk(15)
local_otsu = rank.otsu(img, footprint)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu


#thresh = threshold_local(img, block_size = 10, offset=10)
binary = img >= threshold_global_otsu #img > thresh
from skimage.exposure import adjust_sigmoid

binary = adjust_sigmoid(img, 0.5, 15)

imsave(out_folder + '/blobs.png', binary)
"""
