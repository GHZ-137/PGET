"""
Function of adaptation of the tomogram
"""
from class_Tomography import *
from config import *


# Laplacian of the attenuation graph considering the template type.
# Parameters are bar coo and values
def LapG(blobs, vals):
    n_vals = len(blobs)
    if n_vals == 100: # Square or SVEA /treated similarly
        rows = 10
        cols = 10
        vals = deepcopy(vals)
        vals = np.reshape(vals, (rows, cols))
        res = np.zeros((rows, cols)).astype(np.float)
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                res[r,c] = vals[r,c+1] + vals[r,c-1]\
                           + vals[r+1,c] + vals[r-1,c]\
                           - 4 *vals[r,c]
    if n_vals != 100: # Hexagon with row lengths 7-13-7
        res = []
    
    return res
    

# Adaptation function parameters are tomogram, bar coo and Mu values
def f_adapt(tomo,  blobs, vals, sub_folder = ''):
    # Error
    err = tomo.art(n_iter, poly_order, eta)
    img = tomo.image(out_folder + '/' + sub_folder + '/temp_0.png', "Mu")
    img = tomo.image(out_folder + '/' + sub_folder + '/temp_1.png', "Lambda")
    
    # Simulated sinogram
    ###tomo.project(poly_order)

    # Objectives
    L_UB = a_ref * np.max(tomo.true_sino)/ np.max(tomo.sim_sino)

    L_max = np.max(tomo.Lambda)
    L_ratio = L_max / (L_UB * tomo.n_detect)
    sino_ratio = np.max(tomo.true_sino)/ np.max(tomo.sim_sino)
    
    # Normalized Lam and Mu fit
    L_norm = (tomo.Lambda - np.min(tomo.Lambda)) / (np.max(tomo.Lambda)-np.min(tomo.Lambda))
    L_norm = L_norm * L_UB
    M_norm = (tomo.Mu - min_Mu) / (max_Mu - min_Mu)
    L_norm = np.reshape(L_norm, np.prod(L_norm.shape))
    M_norm = np.reshape(M_norm, np.prod(M_norm.shape))
        
    res = 0
    for c in range(len(L_norm)):
        if L_norm[c] > M_norm[c]:
            # TAKEN OUT M_NORM
            res += L_norm[c] ##+ M_norm[c]
    L_fit = res /tomo.n_detect

    # Regularization Laplacian
    Lap = LapG(blobs, vals)
    if len(Lap)> 0:
        L_Lap = np.sqrt( np.sum(Lap**2) / len(blobs))
    else:
        L_Lap = 0
    #print( np.sort(np.unique(Lap, return_counts = True)[0])[::-1])
    #print(L_Lap)

    
    return [err, L_max, L_ratio, sino_ratio, L_Lap, L_fit]
