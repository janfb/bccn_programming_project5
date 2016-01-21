import numpy as np
from brian import Network, Equations, NeuronGroup, Connection,\
    SpikeMonitor, raster_plot, StateMonitor, clear, reinit
from brian.stdunits import ms, mV
from matplotlib import pylab
from utils import get_cluster_connection_probs
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

pylab.rcParams['figure.figsize'] = 12, 8  # changes figure size (width, height) for larger images

# Options
realizations = 1
trials = 1

# we use 5000 neurons, 4000 e and 1000 i. use alpha to scale it down
alpha = 1 # use only quarter of the neurons

# cluster parameter
REE = 1
# number of clusters
k = int(50*alpha)

# Duration of our simulation
T = 3000*ms

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
N_e = int(4000*alpha) # number of exc neurons
N_i = int(1000*alpha) # number of inh neurons
tau_e = 15*ms  # membrane time constant (for excitatory synapses)
tau_i = 10*ms  # membrane time constant (for inhibitory synapses)
tau_syn_2_e = 3*ms  # exc synaptic time constant tau2 in paper
tau_syn_2_i = 2*ms  # inh synaptic time constant tau2 in paper
tau_syn_1 = 1*ms  # exc/inh synaptic time constant tau1 in paper
V_th = -50 * mV  # firing threshold
V_reset = -65 * mV  # reset potential
refr_period = 5*ms  # absolute refractory period

# scale the weights to ensure same variance in the inputs
wee = 0.024 * 15 * mV * np.sqrt(1./alpha)
wei = 0.014 * 15 * mV * np.sqrt(1./alpha)
wii = -0.057 * 15 * mV * np.sqrt(1./alpha)
wie = -0.045 * 15 * mV * np.sqrt(1./alpha)

# Connection probability
p_ee = 0.2
p_ii = 0.5
p_ei = 0.5
p_ie = 0.5
# determine probs for inside and outside of clusters
p_in, p_out = get_cluster_connection_probs(REE, k, p_ee)

# increase cluster weights if there are clusters
wee_cluster = wee if p_in==p_out else 1.9*wee

for realization in range(realizations):
    # clear workspace to make sure that is a new realization of the network
    clear(True, True)
    reinit()

    # set up new random bias parameter for every type of neuron
    mu_e = V_reset + np.random.uniform(1.1, 1.2, N_e) * (V_th-V_reset) # bias for excitatory neurons
    mu_i = V_reset + np.random.uniform(1.0, 1.05, N_i) * (V_th-V_reset) # bias for excitatory neurons

    # Let's create an equation object from our string and parameters
    model_eqs = Equations(eqs_string)

    # Let's create 5000 neurons
    all_neurons = NeuronGroup(N=N_e+N_i,
                              model=model_eqs,
                              threshold=V_th,
                              reset=V_reset,
                              refractory=refr_period,
                              freeze = True,
                              method='Euler',
                              compile=True)

    # Divide the neurons into excitatory and inhibitory ones
    neurons_e = all_neurons[0:N_e]
    neurons_i = all_neurons[N_e:N_e+N_i]

    # set the bias
    neurons_e.mu = mu_e
    neurons_i.mu = mu_i
    neurons_e.tau = tau_e
    neurons_i.tau = tau_i
    neurons_e.tau_2 = tau_syn_2_e
    neurons_i.tau_2 = tau_syn_2_i
    all_neurons.tau_1 = tau_syn_1

    # set up connections
    C = Connection(all_neurons, all_neurons, 'y')

    # do the cluster connection like cross validation: cluster neuron := test idx; other neurons := train idx
    kf = KFold(n=N_e, n_folds=k)
    for idx_out, idx_in in kf: # idx_out holds all other neurons; idx_in holds all cluster neurons
        # connect current cluster to itself
        C.connect_random(all_neurons[idx_in[0]:idx_in[-1]], all_neurons[idx_in[0]:idx_in[-1]], 'y',
                         sparseness=p_in, weight=wee_cluster)
        # connect current cluster to other neurons
        C.connect_random(all_neurons[idx_in[0]:idx_in[-1]], all_neurons[idx_out[0]:idx_out[-1]], 'y',
                         sparseness=p_out, weight=wee)

    # connect all excitatory to all inhibitory, irrespective of clustering
    C.connect_random(all_neurons[0:N_e], all_neurons[N_e:(N_e+N_i)], 'y', sparseness=p_ei, weight=wei)
    # connect all inhibitory to all excitatory
    C.connect_random(all_neurons[N_e:(N_e+N_i)], all_neurons[0:N_e], 'y', sparseness=p_ie, weight=wie)
    # connect all inhibitory to all inhibitory
    C.connect_random(all_neurons[N_e:(N_e+N_i)], all_neurons[N_e:(N_e+N_i)], 'y', sparseness=p_ii, weight=wii)


    # run this network for some number of trials, every time with different initial values
    for trial in range(trials):
        # set up spike monitors
        spike_mon_e = SpikeMonitor(neurons_e)
        spike_mon_i = SpikeMonitor(neurons_i)

        # set initial conditions
        all_neurons.V = V_reset + (V_th - V_reset) * np.random.rand(len(all_neurons))

        # set up network with monitors
        network = Network(all_neurons, C, spike_mon_e, spike_mon_i)
        spike_mon_i = SpikeMonitor(neurons_e)

        ## Calibration phase
        # run for the first half of the time to let the neurons adapt
        network.run(T/2, report='text')
        # reset monitors to start recording phase
        spike_mon_e = SpikeMonitor(neurons_e)
        spike_mon_i = SpikeMonitor(neurons_i)

        ## Recording phase
        network.run(T/2, report='text')


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