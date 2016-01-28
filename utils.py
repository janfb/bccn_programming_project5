import numpy as np
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt


def get_cluster_connection_probs(REE, k, pee):
    p_out = pee * k / (REE + k - 1)
    p_in = REE * p_out
    return p_in, p_out


def spikes_to_binary(M):
    """
    From SpikeMonitor object it returns a binary numpy array with
    spikes in particular time points.
    :param M: SpikeMonitor object
    :return: numpy matrix with spike times
    """

    try:
        tpnts = np.arange(float(M.clock.start), float(M.clock.end), float(M.clock.dt))
    except AttributeError:
        raise AttributeError("SpikeMonitor doesn't contain any recordings")
    binarr = np.zeros((len(M.spiketimes.keys()), len(tpnts)))
    for k, sp_times in M.spiketimes.items():
        if len(sp_times) == 0:
            continue
        for t_sp in sp_times:
            binarr[k][np.argmin(np.abs(t_sp - tpnts))] = 1
    return binarr


def spikes_counter(M, timewin):
    """
    From SpikeMonitor object it returns a numpy array with spikes counts
    in time windows
    spikes in particular time points.
    :param M: SpikeMonitor object
    :return: numpy matrix with spike times
    """
    try:
        tpnts = np.arange(M.clock.start, M.clock.end + 0.5 * timewin, timewin)
    except AttributeError:
        raise AttributeError("SpikeMonitor doesn't contain any recordings")
    counts = np.zeros((len(M.spiketimes.keys()), len(tpnts) - 1))
    for k, sp_times in M.spiketimes.items():
        if len(sp_times) == 0:
            continue
        for t_sp in sp_times:
            idxs = np.where(tpnts >= t_sp)[0]
            if len(idxs) == 0:
                counts[k][-1] += 1
            else:
                counts[k][idxs[0] - 1] += 1
    return counts



def firing_rates(spike_data, time):
    'Return firing rate for each neuron n from *spike_data*'
    return (spike_data.sum(axis=-1)).flatten() / time

def fano_factor(spike_data):
    """
    Computes Fano factor from matrix *spike_data* of shape (r, k, n, m)
    where *r* - realizations, *k* - nr of trials, *n* - nr of neurons,
    *m* - time steps
    """
    return (np.var(spike_data, axis=1)/np.mean(spike_data, axis=1)).flatten()

def corr_coef(trial_data):
    """
    computes pairwise correlation coefficient form given matrix of trial data
    :param trial_data: matrix of trial data with dimension trials x neurons x timewindows.
    :return: correlation matrix rho
    """

    n_trials = trial_data.shape[0]
    n_neurons = trial_data.shape[1]

    rho = np.zeros((n_neurons, n_neurons))
    cov = np.zeros(n_trials)
    var_factor = np.zeros(n_trials)

    for i in range(n_neurons):
        for j in range(i + 1):
            for t in range(n_trials):
                cov[t] = np.mean(trial_data[t, i, :] * trial_data[t, j, :]) - \
                         np.mean(trial_data[t, i, :]) * np.mean(trial_data[t, j, :])
                var_factor[t] = np.sqrt(np.mean(np.var(trial_data[t, i, :])) * np.mean(np.var(trial_data[t, j, :])))
            rho[i, j] = np.mean(cov) / np.mean(var_factor)
            rho[j, i] = rho[i, j]
    return rho

def corr_coef_new(trial_data):
    """
    computes pairwise correlation coefficient form given matrix of trial data
    With numpy corrcoef instead of explicit calculations.
    :param trial_data: matrix of trial data with dimension trials x neurons x timewindows.
    :return: correlation matrix rho
    """

    n_trials = trial_data.shape[0]
    n_neurons = trial_data.shape[1]

    rho = np.zeros((n_neurons, n_neurons))
    cov = np.zeros(n_trials)
    var_factor = np.zeros(n_trials)

    for i in range(n_neurons):
        for j in range(i + 1):
            for t in range(n_trials):
                cov[t] = np.corrcoef(trial_data[t,i,:], trial_data[t,j,:])[0,1]
            cov_n = cov[~np.isnan(cov)]
            rho[i, j] = np.mean(cov_n)
            rho[j, i] = rho[i, j]
    return rho

def extract_cluster_corr_coef(rho, k=50):
    """
    Extracts correlation values for all clusters form correlation matrix rho.
    :param rho: correlation matrix
    :param k: number of clusters in the network
    :return: correlation matrix for every cluster: k x (neuron_pairs)
    """
    n_neurons = rho.shape[0]
    # determine number of pairs for cluster
    neurons_per_cluster = n_neurons / k
    pairs_per_cluster = neurons_per_cluster * (neurons_per_cluster - 1) / 2.
    cluster_corr_coef = np.zeros((k, pairs_per_cluster))
    for k_idx in range(k):
        # get cluster part from corr_coef matrix
        cluster_cc = rho[(k_idx * neurons_per_cluster):(k_idx + 1) * neurons_per_cluster,
                         (k_idx * neurons_per_cluster):(k_idx + 1) * neurons_per_cluster]
        # save lower triangle of current cluster corr matrix in matrix for all clusters
        cluster_corr_coef[k_idx, :] = get_lower_triangle(cluster_cc)
    # remove nans before returning
    return remove_nans(cluster_corr_coef)


def extract_all_corr_coef(rho):
    """
    Extracts only relevant values from correlation matrix rho
    :param rho: correlation matrix
    :return: 1D array of correlation values
    """
    # get lower triangle of the corr matrix as vector
    rho_vec = get_lower_triangle(rho)
    # remove nans
    return remove_nans(rho_vec)


def extract_random_corr_coef(rho, size):
    """
    Extracts values of random subset of neurons pairs from correlation matrix rho
    :param rho: correlation matrix
    :param size: size of subset
    :return: 1D array of correlation values
    """
    # get indices of random pairs
    idx = np.random.randint(0,rho.shape[0]-1, size=size)
    rho_vec = get_lower_triangle(rho)
    # remove nans
    return remove_nans(rho_vec[idx])


def get_lower_triangle(m):
    """
    Extracts lower triangle of matrix, without diagonal
    :param m: matrix
    :return: lower triangle
    """
    # get lower triangle indices without diagonal (k<0)
    i,j = np.tril_indices(m.shape[0], k=-1)
    # return entries in correlation matrix in array
    return m[i,j]


def remove_nans(m, keep_matrix=False):
    """
    removes nans from a matrix. if keep_matrix is True then nans are set to zero and the matrix is returned. Else,
    a vector with all finite values of the matrix is returned.
    :param m: matrix
    :param keep_matrix: flag for keeping the structure of the matrix
    :return: matrix or vec without nans
    """
    if keep_matrix:
        for i in range(m.shape[0]):
            # get current row and set nans to zero
            tmp = m[i, :]
            tmp[np.isnan(tmp)] = 0
            # replace row in m
            m[i, :] = tmp
    else:
        m = m[np.isfinite(m)]
    return m

def sample_in_cluster(nrns=4000, k=50, picked=20):
    '''
    Reduce number of neurons to *picked* in every of *k* clusters.
    :param nrns: total numer of neurons
    :param k: number of clusters
    :param picked: how many neurons pick from each cluster
    :return: indices of chosen neurons
    '''
    ncl = nrns//k                # neuron in cluster
    assert ncl>picked, "picked is too big"
    nrnnumbers = np.arange(nrns) # indices of all neurons
    idxvec = np.zeros(k*picked)  # vector with new indices
    for i in range(k):
        ix_ = np.random.choice(ncl, picked, replace=False)
        idxvec[i*picked:(i+1)*picked] = nrnnumbers[i*ncl:(i+1)*ncl][ix_]
    return idxvec.astype('int')

def plot_histogram(data1, data2, binwidth, xlabel=''):
    """
    Plot histogram for twp given arrays of  data
    :param data1: first array
    :param data2: second array
    :param binwidth: width of the bins
    :param xlabel: string for xlabel
    :return: no return
    """
    # plot the two array as step histogram
    bins1 = np.arange(min(data1), max(data1) + binwidth, binwidth)
    plt.hist(data1, bins=bins1, align='left', histtype='step')
    plt.axvline(data1.mean(), color='b', linestyle='dashed', linewidth=2)
    bins2 = np.arange(min(data2), max(data2) + binwidth, binwidth)
    plt.hist(data2, bins=bins2, align='left', histtype='step')
    plt.axvline(data2.mean(), color='g', linestyle='dashed', linewidth=2)
    plt.legend(['mean uni', 'mean clus', 'Uniform', 'Clustered'])
    plt.xlabel(xlabel)
    plt.ylabel('Count')
