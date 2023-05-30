"""
Plot of data 
Convergence graph of a single condition.

Jose Luis Rubén García-Zurdo
"""
########################################################
# Inclusions
########################################################
import numpy as np
from functions_plot import *
#from config import *

########################################################
# Variables
########################################################
data_labels = ['CMA-ES', 'Random search'] 
#out_folder1 = out_folder + '/evolution'
#out_folder2 = out_folder + '/random_search'
iterations = 100
#runs = 1

data_folder = './results/'
out_folder = data_folder
names = ["01", "02", "03", "04", "06", "07", "09"]

# Make big data-array
def make_big(data_folder, condition):
    global base
    res = []
    for name in names:
        path = data_folder + '/process_'+ name + '_E2'
        f = open(path + '/' + condition + '/results.txt')
        data = f.readlines()[1:]
        f.close()
    
        # Convert to array
        for cont in range(len(data)):
            data[cont] = data[cont].split()[:]
    
        data = np.array(data).astype(np.float)###[:iterations,:]
        data = data[:530] #Limit longer executions
        res.append(data)
    res = np.array(res)
    #base = res[0,0,4]
    return res

########################################################
# DATA. Read file.
#
########################################################
base = 0.0
data_folder = out_folder
condition = 'results'

if iterations >= 200:
    d_x = 25
else:
    d_x = 10

"""
########################################################
# Plot real values
########################################################
out_f_name = 'real_evolution'
title ='CMA-ES evolution.'
var = '% projection error'

columm_id = 1
avg_data = make_big(data_folder, condition)[:, :, columm_id].mean(axis=0)[:iterations]
avg_data = 100 * (base - avg_data) / base; # Percentage

plot([avg_data],
     range(0, iterations),
     labels = [], #data_labels,
     #err = [best_std, avg_std, worst_std],
     y_sup = 5.,
     y_inf = -50,
     d_x = d_x,
     x_label = 'Generation',
     y_label = var,
     title = title,
     f_name = out_folder + '/' + out_f_name + '.png')
"""
########################################################
# Plot evolution
########################################################
out_f_name = 'percent_evolution'
title ='CMA-ES evolution.'
var = '% projection error'

columm_id = 4
condition = 'evolution'
avg_data = make_big(out_folder, condition)[:, :, columm_id]#[:iterations] #mean(axis=0)[:iterations]
#avg_data = avg_data[:,1:]

avg_data = avg_data[:,:100]

for cont in range(len(avg_data)):
    avg_data[cont] = 100 * (0+ avg_data[cont]) / avg_data[cont][0]
mean1 = np.mean(avg_data, axis=0)
std1 = np.std(avg_data, axis=0)

# Random evolution
columm_id = 4
condition = 'random_search'
avg_data = make_big(out_folder, condition)[:, :, columm_id]#.mean(axis=0)[:iterations]

# Sample longer random evolution
idx = np.around(np.linspace(0, avg_data.shape[1]-1, iterations)).astype(int)
avg_data = avg_data.T
avg_data = avg_data[idx]
avg_data = avg_data.T


avg_data = avg_data[:,:100]

for cont in range(len(avg_data)):
    avg_data[cont] = 100 * (0+ avg_data[cont]) / avg_data[cont][0]
mean2 = np.mean(avg_data, axis=0)
std2 = np.std(avg_data, axis=0)

plot([mean1, mean2],
     range(0, iterations),
     labels = data_labels,
     err = [std1, std2],
     y_sup = 100,
     y_inf = 99.0,
     d_x = d_x,
     x_label = 'Generation',
     y_label = var,
     title = title,
     f_name = out_folder + '/_' + out_f_name + '.png')

dd
########################################################
# Plot differences
########################################################
"""
out_f_name = 'percent_diff_evolution'
title ='CMA-ES best value diff. by generation.'

# Differentiate
avg_data = abs (avg_data1[1:] - avg_data1[0:-1])

plot([avg_data],
     range(0, iterations),
     labels = [], #data_labels,
     #err = [best_std, avg_std, worst_std],
     y_sup = 0.1, #3
     y_inf = 0.,
     d_x = d_x,
     x_label = 'Generation',
     y_label = var,
     title = title,
     f_name = out_folder + '/' + out_f_name + '.png')
"""

