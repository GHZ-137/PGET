"""
Objective 2 - a priori activity-attenuation relation

PGET tomographic reconstruction assuming:
 Detected bars
 [Mu_min, Mu_max] = 0.01 -> 0.14
 Constant Lambda: [L_min, Lam_max] = 0.0 -> a
 Or random Lambda
 Theoretic Lam upper bound = a * max(Sino_real)/ max(Sino_sim)
 Choose "a" than minimizes difference L_max with L_UB => No big difference
 Latter, clip Lam in [0.0, Lam_UB]
 
"""

###############################
# Imports
###############################
from scipy.stats import entropy
from skimage.color import gray2rgb
from class_Tomography import *
from functions_f_adapt import *
from functions_images import *
from functions_plot import *
from config import *

###############################
# Tomography with total uniform attenuation
# to get corrected rotated image to get bars
###############################
np.random.seed(21);
tomo = Tomography()
tomo.read(path, corr_angle_cont, angle_step)
tomo.art(n_iter, poly_order, eta)
#tomo.project(poly_order)
r = np.max(tomo.true_sino) / np.max(tomo.sim_sino)
print("Max ratio:", r)

cv2.imwrite(out_folder + "/true_sino.png", rescale2(tomo.true_sino))
cv2.imwrite(out_folder + "/sim_sino.png", rescale2(tomo.sim_sino))

###############################
# Render an image (with several iterations)
# to get correct angle
###############################
tomo.art(n_iter, poly_order, eta)
img = tomo.image(out_folder + '/corrected.png')

img = cut_zoom(img)
cv2.imwrite(out_folder + '/evolution/Lam_0.png', img)
fin;

###############################
# Obtain blobs with external program.
# Load blobs.
###############################
f = open(out_folder + '/blobs', 'rb')
blobs = pickle.load(f); f.close()

###############################
# Choose "a" value (bars uniform atte.)
# than minimizes the difference L_max, L_UB
###############################
# Attenuation template
# with bars -> a, background -> min_Mu

errs = []
L_maxs = []
L_ratios = []
sino_ratios = []
L_fits = []

# Constant/rnd condition
constant_a = True # True, False 
n_repe = 1 #1, 5
p_f_name = 'img_' # img_, rnd_img

n_bins = 10
norm_std = 0.05

aa = np.linspace(min_Mu, max_Mu, 5)
##################
# Calculate metrics
##################
for a in aa:
    my_errs = [];my_L_maxs = [];my_L_ratios = [];my_sino_ratios = [];my_L_fits = []

    for cont in range(n_repe):
        print(str(a), str(cont))
        
        # Mask
        if constant_a:
            vals = np.array( [a] * len(blobs))
        else:
            vals = np.random.normal(a, norm_std, len(blobs))
            vals = np.clip(vals, min_Mu, max_Mu)
        
        Mu = create_mask(blobs, vals, [tomo.n_detect]*2, bck = min_Mu)
        tomo.Mu = Mu

        # f_adapt
        err, L_max, L_ratio, sino_ratio, L_fit = f_adapt(tomo)
    
        my_errs.append(err);
        my_L_maxs.append(L_max);
        my_L_ratios.append(L_ratio);
        my_sino_ratios.append(sino_ratio);
        my_L_fits.append(L_fit);

    errs.append(np.mean(my_errs))
    L_maxs.append(np.mean(my_L_maxs))
    L_ratios.append(np.mean(my_L_ratios))
    sino_ratios.append(np.mean(my_sino_ratios))
    L_fits.append(np.mean(my_L_fits))
    
# Additional processing
errs = np.array(errs)
L_maxs = np.array(L_maxs)
L_ratios = np.array(L_ratios)
sino_ratios = np.array(sino_ratios)
L_fits = np.array(L_fits)

#Plot function
def my_plot(out_folder, out_f_name, title, yy, xx):
    plot([yy], xx, labels = [],
     y_inf = np.min(yy) - 0.05 * np.min(yy), y_sup = np.max(yy) * 1.05,
     d_x = 1, grid = True,
     x_label ='a', y_label ='',
     title = title, f_name = out_folder + '/' + out_f_name + '.png',
     points = True, dec_x = True)
    return

out_f_name = p_f_name + 'Lam_max'
title ='max(Lambda).'   
my_plot(out_folder, out_f_name, title, L_maxs, aa)

out_f_name = p_f_name + 'L_UB_ratio'
title ='max(Lambda) / L_UB.'   
my_plot(out_folder, out_f_name, title, L_ratios, aa)

out_f_name = p_f_name + 'sino_ratio'
title ='max(true_sino)/max(sim_sino).'   
my_plot(out_folder, out_f_name, title, sino_ratios, aa)

out_f_name = p_f_name + 'Lam_Mu_fit'
title ='Lambda-Mu fit.'   
my_plot(out_folder, out_f_name, title, L_fits, aa)

out_f_name = p_f_name + 'err'
title ='Error.'   
my_plot(out_folder, out_f_name, title, errs, aa)



#L_1 = 2; L_2 = 1 #5;
out_f_name =  p_f_name + 'err_+_L_fit'
title =str(L_1)+' * Error + '+str(L_2)+' * L_fit.'
obj = L_1 * errs + L_2 * L_fits
my_plot(out_folder, out_f_name, title, obj, aa)


