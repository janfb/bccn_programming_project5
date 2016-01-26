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
    'Return firing rate for each neuron n from *spike_data* (n, m)'
    return spike_data.sum(axis=1) / time


def fano_factor(spike_data):
    """
    Computes Fano factor from matrix *spike_data* of shape (k, n, m)
    where *k* - nr of trials, *n* - nr of neurons, *m* - time steps
    """
    if spike_data.ndim > 3:
        return np.var(spike_data.sum(axis=3).mean(axis=2), axis=0) / \
               np.mean(spike_data.sum(axis=3).mean(axis=2), axis=0)
    else:
        return np.var(spike_data.sum(axis=2), axis=0) / \
               np.mean(spike_data.sum(axis=2), axis=0)


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
    pairs_per_cluster = neurons_per_cluster * (neurons_per_cluster - 1) / 2. + neurons_per_cluster
    cluster_corr_coef = np.zeros((k, pairs_per_cluster))
    for k_idx in range(k):
        # get cluster part from corr_coef matrix
        cluster_cc = rho[(k_idx * neurons_per_cluster):(k_idx + 1) * neurons_per_cluster,
                         (k_idx * neurons_per_cluster):(k_idx + 1) * neurons_per_cluster]
        # get entries from lower triangle in array
        i, j = np.triu_indices(cluster_cc.shape[0])  # get indices
        cluster_corr_coef[k_idx, :] = cluster_cc[i, j]
    return cluster_corr_coef


def extract_uniform_corr_coef(rho):
    """
    extracts only relevant values from correlation matrix rho
    :param rho: correlation matrix
    :return: 1D array of correlation values
    """
    # get lower triangle indices
    k, l = np.triu_indices(rho.shape[0])
    # return corr coefs in array
    rho_vec = rho[k, l]
    # remove nans
    return rho_vec[np.isfinite(rho_vec)]


def plot_histogram(data, binwidth):
    """
    Plot histogram for given list of arrays in data
    :param data: list of arrays to be plotted in the histogram
    :param binwidth: width of the bins
    :return: no return
    """
    for d in data:
        # TODO plot mean as triangle
        mean = data[d].mean()
        # TODO fix the access to the list
        bins = np.arange(min(data[d]), max(data[d]) + binwidth, binwidth)
        plt.hist(data[d], bins=bins, align='left', histtype='step')
