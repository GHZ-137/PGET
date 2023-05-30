"""
PGET tomographic reconstruction
"""

###############################
# Imports
###############################
import time
from scipy.stats import entropy
from cmaes import CMA
from skimage.transform import rescale

from class_Tomography import *
from functions_f_adapt import *
from functions_images import *
from config import *

###############################
# Tomography
###############################
# CHANGED RANDOM SEED TO 23
np.random.seed(23)
tomo = Tomography()
tomo.read(path, corr_angle_cont, angle_step) #From config
cv2.imwrite(out_folder + "/evolution/_true_sino.png", rescale2(tomo.true_sino))

###############################
# Load blobs
###############################
f = open(out_folder + '/blobs', 'rb')
blobs = pickle.load(f); f.close()
n_vars = len(blobs)

# Values initialization
#bounds = [min_Mu * 4, max_Mu] # Important to trim lower values IN CONFIG
bounds = np.array( [bounds]* n_vars )

# Constant init values
init_vals = np.array( [init_val] * n_vars)

# Random init values
#init_vals = np.random.normal(a_ref, init_sigma, n_vars) #Normal with center on median
#init_vals = np.clip(init_vals, min_Mu, max_Mu)

# Create mask
Mu = create_mask(blobs, init_vals, [tomo.n_detect]*2, bck = min_Mu)
tomo.Mu = Mu

# Base-line
err = tomo.art(n_iter, poly_order, eta)
img_Lam = tomo.image(out_folder + '/evolution/corrected_baseline.png')
img_Lam = cut_zoom(img_Lam) #, cut = 0.10) #BIGGER
cv2.imwrite(out_folder + '/evolution/Lam_0.png', img_Lam)

#Init activity
#Lam_prev= deepcopy(img_Lam)

# Simulated sinogram and objective
tomo.project(poly_order)
cv2.imwrite(out_folder + "/evolution/_sim_sino.png", rescale2(tomo.sim_sino))

########################
# F_obj
# Complete Lam/Mu fit
########################
res = f_adapt(tomo, blobs, init_vals, sub_folder = 'evolution')
err, L_fit, L_Lap = [res[0]]+[res[-1]]+[res[-2]]
objs = [err, L_fit]
orig_val = L_1 * err + L_2 * L_fit + L_3 * L_Lap
#orig_val = 25
print('Baseline: %.6f\n' % orig_val)
print(objs)


# RESULTS FOLDER
out_folder = out_folder + '/evolution'
### => STOP HERE TO OBTAIN GLOBAL MINIMUM

# Results file
f = open(out_folder + '/results.txt', 'w')
cad = "Gen: \tErr: \t\tobj_1: \t\tobj_2: \t\tBest: \t\tOrig:  \t\tRedu:  \tT: \n"
f.write(cad)

# Baseline, generation 0
tup = (0, orig_val, err, L_fit, orig_val, orig_val, 0, 0)
cad = '%d \t%.6f \t%.6f \t%.6f \t%.6f \t%.6f \t%.2f \t%.2f' % tup
f.write(cad + '\n')

########################
# CMA
########################
my_cma = CMA(init_vals, init_sigma)
my_cma.set_bounds(bounds)
best_val = orig_val
for gen in range(n_gen):
    start = time.perf_counter()
    evs = []
    objs = []
    cont = 0
    for each in range(my_cma.population_size):
        if verbose:
            print(str(cont+1) + '\\' + str(my_cma.population_size))
        cma_vals = my_cma.ask()
        
        Mu = create_mask(blobs, cma_vals, [tomo.n_detect]*2, bck = min_Mu )
        tomo.Mu = Mu
        # Render img
        #err = tomo.art(n_iter, poly_order, eta)
        #img1 = tomo.image(out_folder + '/temp_0.png', "Mu")
        #img2 = tomo.image(out_folder + '/temp_1.png', "Lambda")
     
        ########################
        # F_obj
        res = f_adapt(tomo, blobs, cma_vals, sub_folder = 'evolution')
        err, L_fit, L_Lap = [res[0]]+[res[-1]]+[res[-2]]
        ev = L_1 * err + L_2 * L_fit + L_3 * L_Lap
        print(ev)
        
        # Store vals, fit to recover minimum
        evs.append( (cma_vals, +ev) )
        objs.append( [ev, err, L_fit]) #ev
        cont += 1
        
    my_cma.tell(evs)
    # Recover minimum
    evs = np.array(evs)
    idx = np.argmin(evs[:,1])
    objs = np.array(objs)
    idx_o = np.argmin( [each[0] for each in objs] ) #each[0]*each[1]
    end = time.perf_counter()
    t = end - start
    redu = 100 *( 1- (best_val/orig_val))

    tup = (gen+1, evs[idx,1], objs[idx_o,1], objs[idx_o, 2], best_val, orig_val, redu, t) #0, 1
    
    print('Gen: %d, Err: %.6f, obj_1: %.6f, obj_2: %.6f,  Best: %.6f, Orig: %.6f, Redu.: %.2f, Time: %.2f' % tup)
    cad = '%d \t%.6f \t%.6f \t%.6f \t%.6f \t%.6f \t%.2f \t%.2f' % tup

    # If better (minimization)
    if evs[idx,1] < best_val:
        # Render img
        best_vals, best_val = evs[idx]
        Mu = create_mask(blobs, best_vals, [tomo.n_detect]*2, bck = min_Mu )
        tomo.Mu = Mu

        # Rescale
        img_Mu = rescale2(Mu)
        img_Lam = tomo.image(out_folder + '/temp_1.png', "Lambda")
        
        img1 = cut_zoom(img_Mu) #, cut = 0.10) #BIGGER
        img2 = cut_zoom(img_Lam) #, cut = 0.10) #BIGGER
        cv2.imwrite(out_folder + '/Mu_' + str(gen+1) + '.png', img1)
        cv2.imwrite(out_folder + '/Lam_'+ str(gen+1) + '.png', img2)
         
    cad += '\n'    
    f.write(cad)
f.close()

        
                            
    



