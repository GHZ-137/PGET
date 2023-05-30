"""
Obtain a template best fit to an activity image

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
r = 2.0
n_rows = 10
n_cols = 10
n_intervals = 5

#delta = (12.8/8.48) * 1 * r
#scale = 2.0

def template_square(r, n_rows, n_cols, delta, scale = 1, disp_x = 0, disp_y = 0):
    res = []
    for rr in range(n_rows):
        for c in range(n_cols):
            x = (c * delta) * scale + disp_x
            y = (rr * delta) * scale + disp_y
            res.append([x,y,r])
    
    return res

# Load image
img = imread(out_folder + '/corrected.png')
#img2 = np.copy(img)
#img2 = rescale2(-(-np.exp(img2 / 255.)))

# Bars/blobs as integer coordinates with r = 2
deltas = r * np.linspace(0.8, 1.6, n_intervals*2, endpoint = True, dtype = np.float)
ss = np.linspace(2, 3, n_intervals *2, endpoint = True, dtype = np.float)
dxs = np.linspace(50, 60, n_intervals *2, endpoint = True, dtype = np.float)
dys = np.linspace(50, 60, n_intervals *1, endpoint = True, dtype = np.float)


print("Total: ", str (2 * n_intervals ** 4))

best_fit = -float("Inf")
best_s = 10; best_dx = 0; best_dy = 0; best_cont = 0; best_delta = 0;
cont = 0
for delta in deltas:
    for s in ss:
        for dx in dxs:
            for dy in dxs:
                if cont % 100 == 0:
                    print(str(cont), str(best_cont), best_fit, best_s, best_dx, best_dy)
            
                blobs = template_square(r, n_rows, n_cols, delta, s, dx, dy)
                #blobs = template_hexagon(r, n_rows, n_cols, delta, s, dx, dy)
                
                blobs = np.around(blobs, 0).astype(np.int)

                mask = create_mask(blobs, [255] * len(blobs), img.shape)
                #imsave(out_folder + '/mask'+str(cont)+'.png', mask.astype(np.uint8))
            
                a = np.sum(img * mask)
                b = np.sum(img * (255 - mask))
                fit = float(a) #/ b
                #print (fit)
                if fit > best_fit:
                    best_fit = fit
                    best_s = s
                    best_dx = dx
                    best_dy = dy
                    best_delta = delta
                    best_cont = cont
                    best_blobs = blobs
                cont += 1

blobs = best_blobs
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
imsave(out_folder + '/activity_0.png', act_img)
