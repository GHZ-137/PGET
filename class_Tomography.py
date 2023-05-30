"""
Classes for tomography algebraic reconstruction technique.
"""
import numpy as np
from copy import deepcopy
from skimage.io import *
from skimage.transform import *
from functions_images import *
from functions_lines import *
from config import *

class Tomography:
    def __init__(self):
        self.n_detect = 0
        self.n_angles = 0
        self.center = []
        self.data = []
        self.Lambda = None
        self.Mu = None
        self.error = None
        self.true_sino = None
        self.sim_sino = None
        
    def _print(self):
        print ("Detectors:", self.n_detect)
        print ("Angles:", self.n_angles)
        print ("Centre (pix):" , self.center)
        print ("Pixels:", self.data.shape[0])


    def read(self, path, corr_angle_cont = 0, angle_step = 1):
        global verbose, aperture, min_Mu, max_Mu
        
        self.name =  path[-17:-4]
        data  = np.loadtxt(path)
        self.n_detect = np.int(data[0])
        self.n_angles = np.int(data[1])
        self.center = data[2]
        self.data = data[3:]
        self.angles = np.linspace(0, 360-1, num = self.n_angles, endpoint = True)
        self.lines = lines(self.n_detect, aperture)

        if (self.n_detect * self.n_angles  != self.data.shape[0]):
            print ("Invalid number of pixels")
            os._exit(1)

        if verbose:
            print ("Read ", path)

        # Initial attenuation is set to min_Mu!
        self.Mu = min_Mu * np.ones( [self.n_detect]*2, dtype = np.float32)
        self.Lambda = np.zeros( [self.n_detect]*2, dtype = np.float32)

        # Interleave angles 
        a = []
        for cont in range(360 // 2):
            a.append(cont)
            a.append(359-cont) #a.append(180+cont) #
        self.angles = a

        # Prepare the sinogram
        self.true_sino = np.reshape(self.data, (self.n_angles, self.n_detect))

        # Rotate by displacling sinogram origin
        rot_sino = np.concatenate([self.true_sino [corr_angle_cont:], self.true_sino[:corr_angle_cont]])
        self.true_sino = rot_sino
        if verbose:
            print ("Rotated by ", corr_angle_cont)
            
        # Reduce angles
        red_sino = []
        red_angles = []
        for cont in range(self.n_angles):
            if cont % angle_step == 0:
                red_sino.append(self.true_sino[cont])
                red_angles.append(self.angles[cont])
        self.true_sino = np.array(red_sino)
        self.angles = np.array(red_angles)
        self.n_angles = self.n_angles // angle_step

        # Reorder the sinogram and the data
        #self.true_sino = self.true_sino[np.argsort(self.angles)]
        self.angles = self.angles[ np.argsort(self.angles) ]

        if verbose:
            print ("Reduced angles by ", angle_step)

        if verbose:
            self._print()

        return

    def art(self, n_iter = 3, poly_order = 5, eta = .5, random_eta = []):
        """
        Tomographic reconstruction using ART: projection, error calculation and error substraction
        """
        global verbose

        # Reset
        ######self.Mu = np.ones( [self.n_detect]*2, dtype = np.float32) * 0.01
        self.Lambda = np.zeros( [self.n_detect]*2, dtype = np.float32)
        
        self.poly_order = poly_order
        diffs = []
        cont_iter = 0
        while (cont_iter < n_iter):
            if verbose: print ("Iteration:", cont_iter+1)

            # Reset simulated sinogram
            self.sim_sino = []
    
            Mu_orig = deepcopy(self.Mu)
            angle_prev = 0
            for cont in range(self.n_angles):
                angle =  self.angles[ cont]
                self.Lambda = rotate(self.Lambda, angle - angle_prev, order = poly_order)
                self.Mu = rotate(self.Mu, angle - angle_prev, order = poly_order)
                angle_prev = angle
                
                # Penetrability -> integrating -exp(Mu) by cols.
                Mu_int = np.exp( -1.0 * self.Mu ) #self.Mu = -0.01
                for col in range(self.n_detect -1, 0): Mu_int[:][col] *= Mu_int[:][col+1]
                # Projection -> Lambda * Mu
                # axis = 1 => direction
                q = (self.Lambda * Mu_int).sum(axis = 1)
                q = np.array(q)
                # Store sinogram
                self.sim_sino.append(q)
                
                # Difference
                my_data = self.true_sino[cont]
                diff = q - my_data

                # eta for angle?
                if len(random_eta) > 0:
                    eta_rnd = random_eta[cont]
                else:
                    eta_rnd = eta
                    
                # Aply ART correction
                for row in range(self.n_detect):
                    self.Lambda[row]-=  eta_rnd * diff[row] / self.n_detect
                                   
                self.Lambda = np.clip(self.Lambda, a_min = 0, a_max = np.max(self.Lambda))
                diffs.append(diff / self.n_detect)
                
            cont_iter += 1
            self.Mu = Mu_orig
            self.sim_sino = np.array(self.sim_sino)
      
            # Error is mean of abs or square differences
            #self.error =  np.mean(np.abs(diffs))
            self.error = np.mean( np.power(diffs,2))
                                     
            self.image(out_folder + './intermediate.png')
            if verbose: print (" Error =", self.error)


            # To return RMSE of linear regression
            self.project(poly_order)
            self.error = self.norm_error()
            
        return self.error

    def project(self, poly_order = 5):
        """
        Calculate sinogram projection
        """
        global verbose

        # No reset
        self.poly_order = poly_order
        cont_iter = 0
        while (cont_iter < 1):
            if verbose: print ("Project sinogram.")

            # Reset simulated sinogram
            self.sim_sino = []
            Lam_orig = deepcopy(self.Lambda)
            Mu_orig = deepcopy(self.Mu)
            angle_prev = 0
            for cont in range(0, self.n_angles):        
                angle =  self.angles[ cont]
                
                self.Lambda = rotate(self.Lambda, angle - angle_prev, order = poly_order)
                self.Mu = rotate(self.Mu, angle - angle_prev, order = poly_order)
                
                # Penetrability -> integrating -exp(Mu) by cols.
                Mu_int = np.exp( -1.0 * self.Mu )
                for col in range(self.n_detect -1, 0): Mu_int[:][col] *= Mu_int[:][col+1]
                # Projection -> Lambda * Mu
                # axis = 1 => direction
                q = (self.Lambda * Mu_int).sum(axis = 1)
                self.sim_sino.append(q)
                angle_prev = angle
                
            cont_iter += 1
            self.sim_sino = np.array(self.sim_sino)
            self.Lambda = Lam_orig
            self.Mu = Mu_orig
            
        return

            
    def image(self, path, name = 'Lambda'):
        global min_Mu, max_Mu
        if name == 'Lambda':
            data = deepcopy(self.Lambda) 
            img = rescale2(data,  new_max = 255)
            img = img.T
        
        if name == 'Mu':
           data = deepcopy(self.Mu)
           img = rescale2(data,  minn = min_Mu, maxx = max_Mu)
           #print(np.min(img), np.max(img))
           
           #img = 255 * (data - min_Mu) / (max_Mu - min_Mu)
           ##img = np.uint8(np.round(img, 0))
           
        if (data.any() == None):
            print ("No reconstruction found.")
            return

        cv2.imwrite(path, img)
        return img

    def reset(self):
        # Initial attenuation is set to min_Mu!
        self.Mu = min_Mu * np.ones( [self.n_detect]*2, dtype = np.float32)
        self.Lambda = np.zeros( [self.n_detect]*2, dtype = np.float32)
        return

    def norm_error(self):
        # We must first project to calculate sim_sino!
        # RMSE of (polynomial) linear regression
        
        # Linear regression. Trim sinograms
        sino_w = self.true_sino.shape[1]
        cols = [int(np.around(sino_w * .10)) , int(np.around(sino_w * .90)) ]
        trim_true_sino = self.true_sino[:, cols[0]: cols[1]]
        trim_sim_sino = self.sim_sino[:, cols[0]: cols[1]]
        
        x = trim_sim_sino.flatten()
        x2 = x**2
        x3 = x**3
        y = trim_true_sino.flatten()

        size = np.prod(trim_true_sino.shape)
        
        A = np.vstack([x, x2, x3, np.ones(len(x))]).T #ones
        
        m1, m2, m3, c = np.linalg.lstsq(A, y, rcond=None)[0]
        ##print(m1, m2, m3, c)
    
        y_pred = np.matmul(A,  np.array([m1,m2, m3, c]))
        RMSE = np.sqrt(np.sum((y - y_pred)**2) / size)

        """
        #RMSE of histograms
        bins = 10
        range_ = (0, 5000)
        width = (range_[1] - range_[0])/bins
        f1, _ = np.histogram(y, bins = bins, range = range_)
        f2, _ = np.histogram(y_pred, bins = bins, range = range_)
        f1 = f1 / np.sum(f1)
        f2 = f2 / np.sum(f2)
        RMSE = np.sqrt(np.sum((f1 - f2)**2) / bins)
        """
        return RMSE

