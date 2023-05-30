
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
def f_adapt(tomo, random_eta = [], save = False):
    err = tomo.art(n_iter, poly_order, eta, random_eta)
    img = tomo.image(out_folder + '/temp.png')
    s = snr(img)
    adapt = err - 1*s

    if save:
        img = cut_zoom(img)
        cad = str(cont_gen+1) + '_' + str(np.round(err,2)) + '_' + str(np.round(s,2)) + '.png'
        cv2.imwrite(out_folder + '/' + cad, img )
    
    return [adapt, err, s]

# Variables
out_folder = './_test/genetic'
n_gen = 50
n_ang = 120
n_ind = 12
p_c = 0.8
p_m = 0.2

seeds = np.round(np.random.rand(n_gen) * 100).astype(int)
np.random.seed(21)

prev_best_fit = float('Inf')

# First generation
print("FIRST GENERATION.")
Mu = []
Mu_fit = []
tomo = Tomography()
tomo.read(path, corr_angle_cont, angle_step)

for cont in range(n_ind):
    ind = eta * (.5 +  (np.random.rand(n_ang) * 1.) ) 
    tomo.reset()
    adapt, err, s = f_adapt(tomo, ind)
    
    Mu.append(ind)
    Mu_fit.append ( [adapt, err, s] )
    
# Repeat
for cont_gen in range(n_gen):
    print("\nGENERATION " + str(cont_gen+1))
    Lambda = []
    Lambda_fit = []
# Cross-over
    print("CROSS-OVER.")
    n = int(np.round(p_c * n_ind))
    for cont2 in range(n):
    # Parents
        p1 = int(np.round(np.random.rand()*(n_ind-1)))
        p2 = int(np.round(np.random.rand()*(n_ind-1)))
        p3 = int(np.round(np.random.rand()*(n_ind-1)))
        p4 = int(np.round(np.random.rand()*(n_ind-1)))
        
        f1 = Mu_fit[p1][0]
        f2 = Mu_fit[p2][0]
        f3 = Mu_fit[p2][0]
        f4 = Mu_fit[p2][0]
        
        if f1 <= f2: sel1 = p1
        else: sel1 = p2

        if f3 <= f4: sel2 = p3
        else: sel2 = p4

    # Descendants
        point = int(np.random.rand()*n_ang)
        d1 = np.concatenate( (Mu[sel1][:point], Mu[sel2][point:]) )
        d2 = np.concatenate( (Mu[sel2][:point], Mu[sel1][point:]) )
        Lambda.append(d1)
        Lambda.append(d2)
        
        #tomo = Tomography()
        #tomo.read(path, corr_angle_cont, angle_step)
        tomo.reset()
        adapt, err, s = f_adapt(tomo, d1)
        Lambda_fit.append([adapt, err, s])
        #tomo = Tomography()
        #tomo.read(path, corr_angle_cont, angle_step)
        tomo.reset()
        adapt, err, s = f_adapt(tomo, d2)
        Lambda_fit.append([adapt, err, s])

# Mutate
    print("MUTATION.")
    for cont2 in range(n_ind - n):
        p = int(np.random.rand()*n_ind)
        ind = np.copy(Mu[p])
        for cont in range(len(ind)):
            if np.random.rand() < .5:
                ind[cont] = eta * (.5 +  (np.random.rand() * 1.) ) 
        
        #tomo = Tomography()
        #tomo.read(path, corr_angle_cont, angle_step)
        tomo.reset()
        adapt, err, s = f_adapt(tomo, ind)
        
        Lambda.append(ind)
        Lambda_fit.append([adapt, err, s])

# Elitism
    Mu_fit = np.array(Mu_fit)
    Lambda_fit = np.array(Lambda_fit)
     
    best_Mu = np.argmin(Mu_fit[:,0])
    worst_Lambda = np.argmax(Lambda_fit[:,0])
    
    Lambda[worst_Lambda] = Mu[best_Mu]
    Lambda_fit[worst_Lambda] = Mu_fit[best_Mu]

# Evaluation is done
# Generational replacement
    Mu = Lambda
    Mu_fit = Lambda_fit
    best_Mu = np.argmin(Mu_fit[:,0])
    print("BEST:", Mu_fit[best_Mu])
    
    
    # Save
    if Mu_fit[best_Mu][0] < prev_best_fit:
        tomo.reset()
        f_adapt( tomo, Mu[best_Mu], save = True )
        prev_best_fit = Mu_fit[best_Mu][0]    
        g  = np.where(Mu[np.argmin(Mu_fit[:,0])] > eta)[0].shape[0]
        l = np.where(Mu[np.argmin(Mu_fit[:,0])] < eta)[0].shape[0]
        print(g, '/' ,l)

"""
for cont in range(n):
    np.random.seed(seeds[cont])
    random_eta = eta * (1. +  (np.random.rand(n_ang) * .5) )
    
    tomo = Tomography()
    tomo.read(path, corr_angle_cont, angle_step)
    adapt, err, s = f_adapt(tomo, random_eta)
   
    if adapt < best:
        adapt = np.round(adapt,2);err = np.round(err,2); s = np.round(s,2)
        print(str(cont), " => Reduced. ", str(adapt), str(err), str(s) )
        
        img = tomo.image(out_folder + '/' + str(cont+1) + '_' +str(err)+'_' +str(s)+'.png')
        img = cut_zoom(img)
        cv2.imwrite(out_folder + '/' + str(cont+1) + '_' +str(err)+'_' +str(s)+'.png', img)

        best = adapt
"""
