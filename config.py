"""
PGET tomographic reconstruction configuration.
"""
from functions_images import *

###############################
# Variables and folders
###############################
name = "09"
corr_angle_cont = 315 #4 #315   #165   # Angle correction
L_1 = 1                # Lambdas of objs.
L_2 = 0 ## Sin regularizar. Ya bhy ajuste sinogramas#0, 10
L_3 = 0 ## 0

verbose = True
min_Mu, max_Mu = [0.01, 0.14 * 1.1]
bounds = [min_Mu * 4, max_Mu] # Important to trim lower values

a_ref = 0.1375 #Reference value. Fixed to a value with local minimum f_adapt
init_val = 0.1375 #0.1375

###############################
# Evolution and search
###############################
init_sigma = 0.05 #0.05 when L2       # CMA initial sigma: 0.025 amd 0.05 (conditions)
sigma_rnd_cma = 0.025 #0.025 

use_random_init = False  # Random/constant init values
n_gen = 100              # CMA generations
n_exec = 5               # CMA with random init values repetitions
n_rnd_search_gen = 600 #30*18 # Random search iterations
#L_1 = 1
#L_2 = 5

###############################
# ART and PGET
###############################
#corr_angle_cont = 315   #315   # Angle correction
n_iter = 1              # ART iterations
poly_order = 5          # Interpolation order
angle_step = 3          # Projections decimation factor
eta = 0.25              # ART Learning rate

r = 2.                  # Bars radius
aperture = 1.           # Complete aperture angle (deg.)
pget_d = 360            # PGET tours diameter (mm)

###############################
# Folder and file names
###############################
base = "_training_PGET"
window = "E2"           #E0, E2
condition = "02 - Preprocessed Sinogram Data"
#condition = "03 - Raw Sinogram Data"
use_snr = "" #"preprocessed (low SNR)" #""

# Results folder
out_folder = "./results/process_" + name + "_" + window

if not(file_exists(out_folder)):
    os.mkdir(out_folder)

if not(file_exists(out_folder + '/evolution')):
    os.mkdir(out_folder + '/evolution')

if not(file_exists(out_folder + '/random_search')):
    os.mkdir(out_folder + '/random_search')

path = "../" + base + "/training" + name + "/" + condition + "/"\
       + use_snr + "/training" + name + "_" + window + ".txt"

# Adjust path
if not(file_exists(path)):
    use_snr = ""
    path = "../" + base + "/traning" + name + "/" + condition + "/"\
           + use_snr + "/training" + name + "_" + window + ".txt"  
