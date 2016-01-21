import numpy as np
from brian import Network, Equations, NeuronGroup, Connection, \
    SpikeMonitor, raster_plot, StateMonitor, clear, reinit
from brian.stdunits import ms, mV
from matplotlib import pylab
from utils import get_cluster_connection_probs
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

pylab.rcParams['figure.figsize'] = 12, 8  # changes figure size (width, height) for larger images


def run_simulation(realizations=1, trials=1, t=3000 * ms, alpha=1, ree=1, k=50, verbose=True):
    """Run the whole simulation with the specified parameters. All model parameter are set in the function.

    Keyword arguments:
    :param realizations: number of repititions of the whole simulation, number of network instances
    :param trials: number of trials for network instance
    :param t: simulation time
    :param alpha: scaling factor for number of neurons in the network
    :param ree: clustering coefficient
    :param k: number of clusters
    :param verbose: plotting flag
    :return: numpy matrices with spike times
    """

    # The equations defining our neuron model
    eqs_string = '''
                dV/dt = (mu - V)/tau + x: volt
                dx/dt = -1.0/tau_2*(x - y/tau_1) : volt/second
                dy/dt = -y/tau_1 : volt
                mu : volt
                tau: second
                tau_2: second
                tau_1: second
                '''
    # Model parameters
    n_e = int(4000 * alpha)  # number of exc neurons
    n_i = int(1000 * alpha)  # number of inh neurons
    tau_e = 15 * ms  # membrane time constant (for excitatory synapses)
    tau_i = 10 * ms  # membrane time constant (for inhibitory synapses)
    tau_syn_2_e = 3 * ms  # exc synaptic time constant tau2 in paper
    tau_syn_2_i = 2 * ms  # inh synaptic time constant tau2 in paper
    tau_syn_1 = 1 * ms  # exc/inh synaptic time constant tau1 in paper
    vt = -50 * mV  # firing threshold
    vr = -65 * mV  # reset potential
    refrac = 5 * ms  # absolute refractory period

    # scale the weights to ensure same variance in the inputs
    wee = 0.024 * 15 * mV * np.sqrt(1. / alpha)
    wei = 0.014 * 15 * mV * np.sqrt(1. / alpha)
    wii = -0.057 * 15 * mV * np.sqrt(1. / alpha)
    wie = -0.045 * 15 * mV * np.sqrt(1. / alpha)

    # Connection probability
    p_ee = 0.2
    p_ii = 0.5
    p_ei = 0.5
    p_ie = 0.5
    # determine probs for inside and outside of clusters
    p_in, p_out = get_cluster_connection_probs(ree, k, p_ee)

    # increase cluster weights if there are clusters
    wee_cluster = wee if p_in == p_out else 1.9 * wee

    # define numpy array for data storing
    all_data = np.zeros((realizations, trials, n_e+n_i, t/ms))

    for realization in range(realizations):
        # clear workspace to make sure that is a new realization of the network
        clear(True, True)
        reinit()

        # set up new random bias parameter for every type of neuron
        mu_e = vr + np.random.uniform(1.1, 1.2, n_e) * (vt - vr)  # bias for excitatory neurons
        mu_i = vr + np.random.uniform(1.0, 1.05, n_i) * (vt - vr)  # bias for excitatory neurons

        # Let's create an equation object from our string and parameters
        model_eqs = Equations(eqs_string)

        # Let's create 5000 neurons
        all_neurons = NeuronGroup(N=n_e + n_i,
                                  model=model_eqs,
                                  threshold=vt,
                                  reset=vr,
                                  refractory=refrac,
                                  freeze=True,
                                  method='Euler',
                                  compile=True)

        # Divide the neurons into excitatory and inhibitory ones
        neurons_e = all_neurons[0:n_e]
        neurons_i = all_neurons[n_e:n_e + n_i]

        # set the bias
        neurons_e.mu = mu_e
        neurons_i.mu = mu_i
        neurons_e.tau = tau_e
        neurons_i.tau = tau_i
        neurons_e.tau_2 = tau_syn_2_e
        neurons_i.tau_2 = tau_syn_2_i
        all_neurons.tau_1 = tau_syn_1

        # set up connections
        connections = Connection(all_neurons, all_neurons, 'y')

        # do the cluster connection like cross validation: cluster neuron := test idx; other neurons := train idx
        kf = KFold(n=n_e, n_folds=k)
        for idx_out, idx_in in kf:  # idx_out holds all other neurons; idx_in holds all cluster neurons
            # connect current cluster to itself
            connections.connect_random(all_neurons[idx_in[0]:idx_in[-1]], all_neurons[idx_in[0]:idx_in[-1]], 'y',
                                       sparseness=p_in, weight=wee_cluster)
            # connect current cluster to other neurons
            connections.connect_random(all_neurons[idx_in[0]:idx_in[-1]], all_neurons[idx_out[0]:idx_out[-1]], 'y',
                                       sparseness=p_out, weight=wee)

        # connect all excitatory to all inhibitory, irrespective of clustering
        connections.connect_random(all_neurons[0:n_e], all_neurons[n_e:(n_e + n_i)], 'y', sparseness=p_ei, weight=wei)
        # connect all inhibitory to all excitatory
        connections.connect_random(all_neurons[n_e:(n_e + n_i)], all_neurons[0:n_e], 'y', sparseness=p_ie, weight=wie)
        # connect all inhibitory to all inhibitory
        connections.connect_random(all_neurons[n_e:(n_e + n_i)], all_neurons[n_e:(n_e + n_i)], 'y', sparseness=p_ii,
                                   weight=wii)

        # run this network for some number of trials, every time with different initial values
        for trial in range(trials):
            # set up spike monitors
            spike_mon_e = SpikeMonitor(neurons_e)
            spike_mon_i = SpikeMonitor(neurons_i)

            # set initial conditions
            all_neurons.V = vr + (vt - vr) * np.random.rand(len(all_neurons))

            # set up network with monitors
            network = Network(all_neurons, connections, spike_mon_e, spike_mon_i)

            # Calibration phase
            # run for the first half of the time to let the neurons adapt
            network.run(t / 2, report='text')

            # TODO find a way to reset the monitor properly
            # reset monitors to start recording phase
            #spike_mon_e = SpikeMonitor(neurons_e)
            #spike_mon_i = SpikeMonitor(neurons_i)

            # Recording phase
            #network.run(t / 2, report='text')

            # TODO save the spike monitor output to the all_data matrix.
            #all_data[realization, trial, :, :] =

    if verbose:
        # Plot spike raster plots, blue exc neurons, red inh neurons
        plt.figure()
        plt.subplot(211)
        raster_plot(spike_mon_i, color='r')
        plt.title('Inhibitory neurons')
        plt.subplot(212)
        raster_plot(spike_mon_e)
        plt.title('Excitatory neurons')

        # Show the plots
        plt.show()

    return all_data

results = run_simulation()