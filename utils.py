import numpy as np

def get_cluster_connection_probs(REE, k, pee):
    p_out = pee * k/(REE + k -1)
    p_in = REE * p_out
    return p_in, p_out

def spikes_to_binary(M):
    '''
    From SpikeMonitor object it returns a binary numpy array with
    spikes in particular time points.
    :param M: SpikeMonitor object
    :return: numpy matrix with spike times
    '''
    try:
        tpnts = np.arange(float(M.clock.start), float(M.clock.end), float(M.clock.dt))
    except AttributeError:
        raise AttributeError("SpikeMonitor doesn't contain any recordings")
    binarr = np.zeros((len(M.spiketimes.keys()), len(tpnts)))
    for k, sp_times in M.spiketimes.items():
        if len(sp_times)==0:
            continue
        for t_sp in sp_times:
            binarr[k][np.argmin(np.abs(t_sp-tpnts))] = 1
    return binarr
