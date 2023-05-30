
###############################
# Imports
###############################
from scipy.stats import entropy
from skimage.color import gray2rgb
from class_Tomography import *
#from functions_f_adapt import *
from functions_images import *
from functions_plot import *
from config import *

###############################
# Tomography with total uniform attenuation
# to get corrected rotated image to get bars
###############################
n = 900
n_ang = 120
out_folder = './_test/random'
seeds = np.round(np.random.rand(n) * 100).astype(int)

def f_adapt(tomo, random_eta = []):
    err = tomo.art(n_iter, poly_order, eta, random_eta)
    img = tomo.image(out_folder + '/temp.png')
    s = snr(img)
    adapt =  -s #err - 1*s
    return [adapt, err, s]

# Baseline
"""
tomo = Tomography()
tomo.read(path, corr_angle_cont, angle_step)
adapt, err, s = f_adapt(tomo)
print(adapt)
err = np.round(err,2); s = np.round(s,2)
img = tomo.image(out_folder + '/_baseline_' +str(err)+'_' +str(s)+'.png')
img = cut_zoom(img)
cv2.imwrite(out_folder + '/_baseline_' +str(err)+'_' +str(s)+'.png', img)
fin
"""

best = 6.1
for cont in range(n):
    np.random.seed(seeds[cont])
    random_eta = eta * (.5 +  (np.random.rand(n_ang) * 1.) )
    
    tomo = Tomography()
    tomo.read(path, corr_angle_cont, angle_step)
    adapt, err, s = f_adapt(tomo, random_eta)
   
    if adapt < best:
        adapt = np.round(adapt,2);err = np.round(err,2); s = np.round(s,2)
        print(str(cont), " => Reduced. ", str(adapt), str(err), str(s) )
        print(np.mean(random_eta))
        img = tomo.image(out_folder + '/' + str(cont+1) + '_' +str(err)+'_' +str(s)+'.png')
        img = cut_zoom(img)
        cv2.imwrite(out_folder + '/' + str(cont+1) + '_' +str(err)+'_' +str(s)+'.png', img)

        best = adapt

