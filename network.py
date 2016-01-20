from brian import Network, Equations, NeuronGroup, Connection,\
    SpikeMonitor, raster_plot, StateMonitor, clear, reinit
from brian.stdunits import ms, mV
from matplotlib import pylab

pylab.rcParams['figure.figsize'] = 12, 8  # changes figure size (width, height) for larger images

clear(True, True)
reinit()  # To reinit BRIAN clocks and remove all old BRIAN objects from namespace,
# it's usually a good idea to put this at the beginning of a script

# The equations defining our neuron model
eqs_string = '''
            dV/dt = (mu - V + x)/tau: volt
            dx/dt = -x/tau_2 + y/tau_1 : volt
            dy/dt = -y/tau_1 : volt
            mu : volt
            tau: ms
            tau_2: ms
            tau_1: ms
            '''