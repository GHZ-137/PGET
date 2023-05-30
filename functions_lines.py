"""
Line-tracing between voxels and sensors in PGET
"""

# Python imports
import pickle
import numpy as np
from skimage.draw import line_aa
import cv2 as cv2

# Variables


# Return a ordered list of coord. and int. values of line pixels for
# each one of the sensors.
#
# Values should be multiplied by attenuation Mu and
# integrated bottom-to-top to give the total attenuation exponent
# of each pixel in each line of the visibility triangle for each sensor.
# Square unitary pixels are assumed.
#
# Emissivity for each pixel in the triangle for each sensor should be multipled
# by negative exponential of total attenuation exponent.

def lines(n_sensors, aperture, flip = True):
    # Aperture trigonometry -> Top width of triangle
    top_w = n_sensors * np.sin(0.5 * np.pi * aperture / 180.)
    top_w = int(np.round(top_w, 0))
    if (top_w % 2) == 1:
        top_w -= 1

    #print(top_w)
    lines = []
    # For each sensor, discarding outer values
    for n in range(0, n_sensors):
        l = []
        # For each line to the sensor
        for top_col in range(n-top_w//2, n+top_w//2):
            if 0 <= top_col <n_sensors:
                res = line_aa(0, top_col, n_sensors-1, n)
                # Stack values to array
                res = np.column_stack((res[0], res[1], res[2]))
                # Flip it!
                if flip: res = np.column_stack((res[1], res[0], res[2]))
                # Integer coordinates
                res[:,:2] = np.round(res[:,:2], 0)
                # Sort top_to-bottom
                idx = np.argsort(res[:,0]) ##[::-1]
                res = res[idx]

                """
                # Subsort left->right according to triangle half
                for cont_id in range(n_sensors):
                    idx_row = np.where(res[:,0] == cont_id)[0]
                    if len(idx_row) > 1:
                        idx_col = np.argsort(res[idx_row][:,1])
                        if top_col - n < 0: #Left half
                            idx_col = idx_col[::-1]
                        res[idx_row] = res[idx_row][idx_col]
                """     
                l.append(res)
        lines.append(l)    
    return lines

# Tests
#k = lines(172, 40.)

#for each in range(0, 172): print(len(k[each]))
#for each in k[0][1]: print(each)
#for each in k[-1][-1]: print(each)

#res = k[0][10]
#for each in res: print(each)
