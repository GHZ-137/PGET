"""
Auxiliary functions for files and image processing
"""

# Python imports
import pickle, os, glob
from shutil import copyfile
from copy import deepcopy
import numpy as np
from skimage.draw import disk, circle_perimeter, circle_perimeter_aa
import matplotlib.pyplot as plt
import cv2 as cv2

#######################################################
# Files and output
#
#######################################################
def to_2f(n):
    if type(n) == list or type(n) == np.ndarray:
        res = []
        for n2 in n:
            res.append(to_2f(n2))
        return res
    return (float( int(n*100) )/100)

def file_exists(path):
    """
    Check if file exists in certain path
    """
    res = os.path.exists(path) 
    
    return res

def delete_file(path):
    """
    Deletes a file
    """
    os.remove(path)

def files(path, mask = "*.*", numeric = False):
    """
    Return the sorted file names in a path with a given extension.
    If numeric, perform numeric ordering of files.
    """
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    
    num_key = lambda x: int( file_name( x ).split(".",1)[0])
    
    res =  glob.glob(os.path.join(path, mask))
    if numeric:
        res.sort(key = lambda x: int( file_name( x ).split(".",1)[0]) )            
    else:
        res.sort
        
    return res

def erase_files(path):
    """
    Remove files in a directory
    """
    ff = files(path)
    for f in ff:
        os.remove(f)
    return

def file_name(path):
    """
    Return the name of a file from its complete path.
    """
    _, name = os.path.split(path)
    return name

def file_number(path):
    """
    Return the next number file name from existing numeric files in a path
    """
    n = files(path)
    if n == []:
        return "0"
    else:
        n = [ each.split(".",1)[0] for each in n]
        n = [int(each.split("\\",1)[-1]) for each in n]
        n = sorted(n)
        n = n[-1] + 1
        return str(n)

def show(images, sequential = True):
    """
    Show a image or list of images consecutively.
    """
    if images == []:
        print ("No image(s) to show")
        return

    if type(images) == list:
        #or ( type(images) == np.ndarray and len(images.shape) > 3):
        if not sequential:
            cont = 0
            for img in images:
                cv2.imshow('image ' + str(cont),img)
                cont += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cont = 0
            for img in images:
                cv2.imshow('image ' + str(cont),img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cont += 1

    else:
        cv2.imshow('image ',images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def rescale2(img, new_min = 0., new_max = 255., levels = 8, minn = None, maxx = None):
    """
    Rescale values of a image
    """
    # color images
    if len(img.shape) == 3:
        minn1 = np.min(img[:,:,0]); minn2 = np.min(img[:,:,1]); minn3 = np.min(img[:,:,2])
        maxx1 = np.max(img[:,:,0]); maxx2 = np.max(img[:,:,1]); maxx3 = np.max(img[:,:,2])
        minn = np.min( minn1, minn2, minn3)
        maxx = np.max( maxx1, maxx2, maxx3)
        res = new_min + ( (np.float32(img)-np.min(img)) / (np.max(img)-np.min(img)) ) * (new_max - new_min)
    else:
        if minn == None or maxx == None:
            minn = np.min(img)
            maxx = np.max(img)
            
        res = minn + ( (np.float32(img)-minn) / (maxx -minn) ) * (new_max - new_min)

    if levels == 8:
        return np.uint8(np.round(res,0))

    if levels == 16:
        return np.uint16(np.round(res,0))

def resize_max(img, size, method = cv2.INTER_AREA):
    """
    Resize max dimension of an image to size
    """
    h,w = img.shape[:2]
    if h > w:
        k = size[0] / float(h)
    else:
        k = size[1] / float(w)
    if k < 1:
        interpolation = method
    else:
        interpolation = method
    size = ( int(round(img.shape[1] * k)), int(round(img.shape[0] * k) ) )
    img = cv2.resize(img, size, interpolation = interpolation )
    return img

#######################################################
# Mask
#
#######################################################
# Overlap circles to color image
def create_circles(blobs, img):
    for blob in blobs:
        y, x, r = blob
        rr, cc  = circle_perimeter(y, x, r, shape = img.shape)
        img[rr,cc, 0] = 255
    return img

# Create mask from a blob array and a list of values
def create_mask(blobs, values, img_shape, bck = 0., inc_r = 0.25 ):
    mask = bck * np.ones(img_shape).astype(np.float32)
    for cont in range(len(blobs)):
        y, x, r = blobs[cont]
        rr, cc = disk((y, x), r + inc_r, shape = img_shape)
        mask[rr,cc] = values[cont]

        cont += 1
    return mask

#######################################################
# Add data to images and crop.
#
#######################################################
# Add size lectures to the margins
# Parameters are pairs of (min, max)  values and n. of ticks
def add_borders(img, v_limits, h_limits, n=4):
    marg = 30; padd = 10
    if len(img.shape) == 2:
        res = 255 * np.ones( [img.shape[0] + marg+padd, img.shape[1] + marg+padd])
        res[padd:img.shape[0]+padd, marg:-padd] = img[:,:]
        color = 0
    else:
        res = 255 * np.ones( [img.shape[0] + marg+padd, img.shape[1] + marg+ padd, 3])
        res[padd:img.shape[0]+padd, marg:-padd, :] = img[:,:,:]
        color = [0,0,0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35

    # Vertical numbers
    vv = np.linspace(v_limits[0], v_limits[1], n, endpoint = True, dtype = np.int)
    row = 0 + padd; col = 5;
    for v in vv:
        cv2.putText(res, str(v),(col, row+3), font, font_scale * 1.,(0), 1)
        cv2.line(res, (marg-3, row), (marg, row), color = color)
        row +=  img.shape[0] // (n-1)
    cv2.line(res, (marg, padd), (marg, img.shape[0]+padd), color = color)
    
    # Horizontal numbers
    hh = np.linspace(h_limits[0], h_limits[1], n, endpoint = True, dtype = np.int)
    row = img.shape[0]+padd; col =  marg;
    for h in hh:
        cv2.putText(res, str(h),(col-10, row+15), font, font_scale * 1.,(0), 1)
        cv2.line(res, (col, row+3), (col, row), color = color)
        col +=  img.shape[1] // (n-1)
    cv2.line(res, (marg, img.shape[0]+padd), (marg+img.shape[1], img.shape[0]+padd), color = color)
    
    return res


# Add a color scale of values to the right margin
# Parameters are an array of values, n. of ticks and n. of decimals
def add_scale(img, values, n=4, dec = 0):
    marg_old = 30;
    marg = 50; padd = 10; off = 20
    if len(img.shape) == 2:
        res = 255 * np.ones( [img.shape[0], img.shape[1] + marg+padd])
        res[:img.shape[0], :img.shape[0]] = img[:,:]
        color = 0
    else:
        res = 255 * np.ones( [img.shape[0], img.shape[1] + marg+padd, 3])
        res[:img.shape[0], :img.shape[1],:] = img[:,:,:]
        color = [0,0,0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35

    v_limits = [np.min(values), np.max(values)]
    # Vertical numbers
    vv = np.linspace(v_limits[0], v_limits[1], n, endpoint = True, dtype = np.float)[::-1]
    row = 0 + padd; col = img.shape[1] + 8 + off;
    for v in vv:
        cv2.putText(res, ("%.2f" % (v)),(col, row+3), font, font_scale * 1.,(0), 1)
        
        cv2.line(res, (img.shape[1]+off+3, row), (img.shape[1]+off, row), color = color)
        row +=  (img.shape[0] - marg_old - padd) // (n-1)
        
    cv2.line(res, (img.shape[1]+ off, padd), (img.shape[1] + off, img.shape[0]-marg_old), color = color)

    # Color code
    vv = np.linspace(v_limits[0], v_limits[1], img.shape[0]-marg_old - padd, endpoint = True, dtype = np.float)
    row = 0 + padd; #col = img.shape[1] - 5;
    #print(vv)
    for v in vv[::-1]:
        v = (v - v_limits[0]) / (v_limits[1]-v_limits[0])
        color = [ int(255*v), 10, int(255*(1.-v))]
        cv2.line(res, (img.shape[1]-10 +1, row), (img.shape[1]+off, row), color = color)
        row +=  1
        


    return res

# Get the average value of a list of blobs in a image
def get_levels(img, blobs):
    res = np.zeros(len(blobs))
    for cont in range(len(blobs)):
        # Changed y, x order
        y, x, r = blobs[cont]
        rr, cc = disk((y, x), r, shape = img.shape)
        res[cont] = np.mean(img[rr,cc])
    return res

# Crop an image given its blobs  and r
# Return cropped image, cropped blobs and limit coo. in given units
def crop(img, blobs, pget_d, r):
    blobs = deepcopy(blobs)
    min_x = int(np.min(blobs[:,1]) -np.ceil(r))
    max_x = int(np.max(blobs[:,1]) +np.ceil(r))
    min_y = int(np.min(blobs[:,0]) -np.ceil(r))
    max_y = int(np.max(blobs[:,0]) +np.ceil(r))

    borders = [ [pget_d * min_x /float(img.shape[1]),
                 pget_d * max_x /float(img.shape[1])],
                [pget_d * min_y /float(img.shape[0]),
                 pget_d * max_y /float(img.shape[0])] ]


    img = img[min_y:max_y + 1, min_x:max_x + 1]
    blobs[:,1] = blobs[:,1] - min_x
    blobs[:,0] = blobs[:,0] - min_y

    return [img, blobs, borders]

def activity_image(blobs, values, pget_d, r, limits = None):
    blobs = deepcopy(blobs)
    min_x = int(np.min(blobs[:,1]) -r)
    max_x = int(np.max(blobs[:,1]) +r)
    min_y = int(np.min(blobs[:,0]) -r)
    max_y = int(np.max(blobs[:,0]) +r)

    img = 255 *np.ones([max_y-min_y, max_x-min_x, 3])
    blobs[:,1] = blobs[:,1] - min_x
    blobs[:,0] = blobs[:,0] - min_y

    if limits != None:
        minn = limits[0]
        maxx = limits[1]
    else:
        minn = np.min(values)
        maxx = np.max(values)
        

    for cont in range(len(values)):
        b = blobs[cont]
        v = (values[cont] - minn) / (maxx - minn)

        y, x, r = b

        rr, cc = disk((y, x), r, shape = img.shape)
        img[rr,cc,:] = [ int(255*v), 10, int(255*(1.-v))]

        rr, cc  = circle_perimeter(y, x, r, shape = img.shape)
        #for cont2 in range(len(rr)):
        #    img[rr[cont2],cc[cont2], :] = [255*(1.-vals[cont2])] * 3
        img[rr,cc,:] = [0, 0, 0]
                
    return img

def cut_zoom(img, cut = 0.25, scale = 2.):
    res = np.copy(img)
    h, w = res.shape
    res = res[ int(cut * h):  h - int(cut * h), int(cut * w):w - int(cut * w)]
    res = resize_max(res, np.array(res.shape) * scale)    
    return res

#######################################################
# Image quality metric
#
#######################################################
def cnr(img, cut = 0.25):
    img2 = np.copy(img)
    h, w = img.shape
    img2 = img2[ int(cut * h):  h - int(cut * h), int(cut * w):w - int(cut * w)]
    res = abs(np.max(img2) - np.min(img2)) / np.std(img2)
    return res

def psnr(img, cut = 0.25):
    img2 = np.copy(img)
    h, w = img.shape
    img2 = img2[ int(cut * h):  h - int(cut * h), int(cut * w):w - int(cut * w)]
    res = 20 *np.log10(np.max(img2) / np.std(img2))
    return res

def snr(img, cut = 0.25):
    img2 = np.copy(img)
    h, w = img.shape
    img2 = img2[ int(cut * h):  h - int(cut * h), int(cut * w):w - int(cut * w)]
    res = np.mean(img2) / np.std(img2)
    return res
