import numpy as np
from brian import Network, Equations, NeuronGroup, Connection,\
    SpikeMonitor, raster_plot, StateMonitor, clear, reinit
from brian.stdunits import ms, mV
from matplotlib import pylab, gridspec

import matplotlib.pyplot as plt

pylab.rcParams['figure.figsize'] = 12, 8  # changes figure size (width, height) for larger images

clear(True, True)
reinit()  # To reinit BRIAN clocks and remove all old BRIAN objects from namespace,
# it's usually a good idea to put this at the beginning of a script

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
# Our model parameters
tau_e = 15*ms # membrane time constant (for excitatory synapses)
tau_i = 10*ms # membrane time constant (for inhibitory synapses)
tau_syn_2_e = 3*ms # exc synaptic time constant tau2 in paper
tau_syn_2_i = 2*ms # inh synaptic time constant tau2 in paper
tau_syn_1 = 1*ms # exc/inh synaptic time constant tau1 in paper
V_th = -50*mV # firing threshold
V_reset = -65*mV # reset potential
refr_period = 5*ms # absolute refractory period
wee = 0.024 * 15*mV
wei = -0.045 * 15*mV
wii = -0.057 * 15*mV
wie = 0.014 * 15*mV

# Number of neurons
N_e = 4000 # number of exc neurons
N_i = 1000 # number of inh neurons

# Our parameters for the bias mu
mu_e = np.random.uniform(1.1, 1.2, N_e) * (V_th-V_reset) # bias for excitatory neurons
mu_i = np.random.uniform(1.0, 1.05, N_i) * (V_th-V_reset) # bias for excitatory neurons

# Connection probability
conn_prob_ee = 0.2
conn_prob_ii = 0.5
conn_prob_ei = 0.5
conn_prob_ie = 0.5

# Duration of our simulation
duration = 2000*ms

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

Cee = Connection(neurons_e, neurons_e, 'y', sparseness=conn_prob_ee, weight=wee)
Cei = Connection(neurons_e, neurons_i, 'y', sparseness=conn_prob_ei, weight=wei)
Cie = Connection(neurons_i, neurons_e, 'y', sparseness=conn_prob_ie, weight=wie)
Cii = Connection(neurons_i, neurons_i, 'y', sparseness=conn_prob_ii, weight=wii)

spike_mon_e = SpikeMonitor(neurons_e)
spike_mon_i = SpikeMonitor(neurons_i)
state_mon_v_e = StateMonitor(neurons_e, 'V', record=[0,1,2])
state_mon_v_i = StateMonitor(neurons_i, 'V', record=[0,1])

all_neurons.V = V_reset + (V_th - V_reset) * np.random.rand(len(all_neurons))

network = Network(all_neurons, Cee, Cei, Cie, Cii, spike_mon_e, state_mon_v_e)
network.add(spike_mon_i, state_mon_v_i)

network.run(duration, report='text')

# Plot spike raster plots, blue exc neurons, red inh neurons
plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
plt.subplot(gs[0])
raster_plot(spike_mon_i, color='r')
plt.title('Inhibitory neurons')
plt.subplot(gs[1])
raster_plot(spike_mon_e)
plt.title('Excitatory neurons')

# Show the plots
plt.show()

