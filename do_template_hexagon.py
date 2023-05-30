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
r = 2. #2
n_cols = 8 #7
n_max_cols = 15 #13
n_intervals = 5

def template_hexagon(r, n_cols, n_max_cols, delta, alpha, scale = 1, disp_x = 0, disp_y = 0):
    # Primera fila: 7
    # Segunda fla: 8
    # ...
    # Hasta 13
    # Respetando h = (raiz(3)/2) * delta
    d2 = delta/2.
    h = (np.sqrt(3)/2) * delta

    
    res = []
    inc_cols = [0,1,2,3,4,5,6, 7,6, 5,4,3,2,1,0] #7,6
    for rr in range( len(inc_cols)):
        for c in range(n_cols + inc_cols[rr]):
            
            if rr <=6:
                x = (float(0) - d2*rr + delta * c) * scale + disp_x
                #x = np.floor(x)
            else:
                x = (float(0) + d2*rr - (delta * (n_cols-1)) + delta *c) * scale + disp_x
                #x = np.ceil(x)
                
            y = (rr * h) * scale + disp_y

            """
            if rr <= n_rows//2:
                y = np.floor(y)
            if rr >= np.ceil(n_rows/2):
                y = np.ceil(y)    
            """
            if x>=0 and y >= 0:
                res.append([y,x,r])
                
    res = np.array(res)   
    # Rotation
    yy = res[:,0]
    xx = res[:,1]
    c_y = np.mean(yy) #.5 * (np.max(yy) + np.min(yy))
    c_x = np.mean(xx) #.5 * (np.max(xx) + np.min(xx))
    alpha = np.pi * (alpha / 180.)
            
    res2 = []
    for y, x, _ in res:
        R = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
        x = x - c_x
        y = y - c_y
        x2, y2 = np.matmul(R, np.array([x,y]).T) + np.array([c_x, c_y]).T
        res2.append([y2, x2, r])
        #x +=c_x
        #y +=c_y
        #print(x,y, x2, y2)
    
    return res2


# Load image
img = imread(out_folder + '/corrected.png')
img2 = np.copy(img)
img2 = rescale2(-(-np.exp(img2 / 255.)))

# Bars/blobs as integer coordinates with r = 2
deltas = 2.1 * r * np.linspace(0.8, 1.6, n_intervals *2, endpoint = True, dtype = np.float)
ss = np.linspace(2, 3, n_intervals *2, endpoint = True, dtype = np.float)
dxs = np.linspace(40, 80, n_intervals *1, endpoint = True, dtype = np.float)
dys = np.linspace(40, 80, n_intervals *1, endpoint = True, dtype = np.float)

ss = np.linspace(1.5, 1.7, 3)
deltas = np.linspace(3.5, 3.9, 3)

print("Total: ", str (4 * n_intervals ** 4))

best_fit = -float("Inf")
best_s = 10; best_dx = 0; best_dy = 0; best_cont = 0; best_delta = 0;
cont = 0
for delta in deltas:
    for alpha in [-.2]: #,0,.75]:
        for s in ss:
            for dx in dxs:
                for dy in dxs:
                    if cont % 100 == 0 and cont > 1:
                        print(str(cont), str(best_cont), best_fit, best_s, best_dx, best_dy)
                        #imsave(out_folder + '/mask'+str(cont)+'.png', mask.astype(np.uint8))
                    
                    blobs = template_hexagon(r, n_cols, n_max_cols, delta, alpha, s, dx, dy)
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

# Post-process
blobs = np.array(blobs)
blobs[:,0] += -1
blobs[:,1] += -2


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
act_img = add_scale(act_img, levels / 255., n=7, dec =2)
imsave(out_folder + '/activity_0.png', act_img)

