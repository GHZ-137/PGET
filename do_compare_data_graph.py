"""
Obtain main plot and data
Convergence graph of a single condition.

Jose Luis Rubén García-Zurdo
"""
########################################################
# Inclusions
########################################################
import numpy as np
from functions_plot import *
####from config import *

########################################################
# Variables
########################################################
data_labels = ['CMA-ES', 'Random search']
iterations = 100

data_folder = "./results/process_" #+ name + "_" + window
names = ["01", "02", "03", "04", "06", "07", "09"]
window = "E2"
out_folder = "./results/"

# Make big data-array
def make_big(data_folder, condition):
    global base, names, window, iterations
    res = []
    for name in names:
        f_name = data_folder + name + "_" + window + '/' + condition
        f = open(f_name + '/results.txt')
        data = f.readlines()[1:]
        f.close()

        # Convert to array
        for cont in range(len(data)):       
            data[cont] = data[cont].split()[:]
        data = np.array(data).astype(np.float)

        # Resample longer data
        idx = np.linspace(0, len(data), iterations, False, dtype = np.int)
        data = data[idx]
        #print(data.shape)
        res.append(data)
    res = np.array(res)
    #base = res[0,0,4]
    return res

########################################################
# Read data and average
#
########################################################
base = 0.0
if iterations >= 200:
    d_x = 25
else:
    d_x = 10

out_f_name = 'averages'
title ='CMA-ES vs. Random Search.'
var = '% projection error'

# Plot multi
columm_best = 4
condition = 'evolution'
avg_data = make_big(data_folder, condition)[:,:, columm_best][:iterations]
for cont in range(len(avg_data)):
    base = avg_data[cont][0]
    avg_data[cont] = 100 * (0+ avg_data[cont]) / base; # Percentage
avg_data1 = np.copy(avg_data)


condition = 'random_search'
avg_data = make_big(data_folder, condition)[:,:, columm_best][:iterations]
for cont in range(len(avg_data)):
    base = avg_data[cont][0]
    avg_data[cont] = 100 * (0+ avg_data[cont]) / base; # Percentage
avg_data2 = np.copy(avg_data)

plot_multi([avg_data1, avg_data2],
     range(0, iterations),
     labels = data_labels,
     #err = [best_std, avg_std, worst_std],
     y_sup = 100.,
     y_inf = 99.,
     d_x = d_x,
     x_label = 'Generation',
     y_label = var,
     title = title,
     f_name = out_folder + '/Multi_' + out_f_name + '.png')


# Plot average
columm_best = 4
condition = 'evolution'
avg_data = make_big(data_folder, condition)[:,:, columm_best].mean(axis=0)[:iterations]
base = avg_data[0]
avg_data1 = 100 * (0+ avg_data) / base; # Percentage

condition = 'random_search'
avg_data = make_big(data_folder, condition)[:,:, columm_best].mean(axis=0)[:iterations]
base = avg_data[0]
avg_data2 = 100 * (0+ avg_data) / base; # Percentage

plot([avg_data1, avg_data2],
     range(0, iterations),
     labels = data_labels,
     #err = [best_std, avg_std, worst_std],
     y_sup = 100.,
     y_inf = 99.,
     d_x = d_x,
     x_label = 'Generation',
     y_label = var,
     title = title,
     f_name = out_folder + '/' + out_f_name + '.png')


for each in avg_data2: print(each)

