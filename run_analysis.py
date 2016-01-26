import numpy as np
import matplotlib.pyplot as plt
import utils

# load the data
all_data_uniform = np.load('uniform_r12t9.npy')
all_data_clustered = np.load('clustered_r12t9.npy')

# load corr coefs
rho_uniform = np.load('rho_uniform_1000.npy')
rho_clustered = np.load('rho_clustered_1000.npy')

# get vector representation of uniform corr coef
rho_uni_vec = utils.extract_all_corr_coef(rho_uniform)
rho_clus_vec = utils.extract_all_corr_coef(rho_clustered)

# get matrix of corr coefs per cluster
rho_per_cluster = utils.extract_cluster_corr_coef(rho_clustered)
rho_uniform_random = utils.extract_random_corr_coef(rho_uniform, size=len(rho_per_cluster))

# plot histogram
plt.ion()
utils.plot_histogram(rho_uniform_random, rho_per_cluster, binwidth=0.01, xlabel='Correlation same cluster')