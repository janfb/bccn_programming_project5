def get_cluster_connection_probs(REE, k, pee):
    p_out = pee * k/(REE + k -1)
    p_in = REE * p_out
    return p_in, p_out