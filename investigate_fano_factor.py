from network import run_simulation
import numpy as np
import matplotlib.pyplot as plt
from brian.stdunits import ms, mV

ree_steps = np.linspace(1, 4, 7)
ff = np.zeros_like(ree_steps)
nnans = np.zeros_like(ree_steps)

for i, ree in enumerate(ree_steps):
    # run simulation with one network, nine trials
    data = run_simulation(trials=4, ree=ree, verbose=False, winlen=100 * ms)
    # take only excitatory neurons
    data = data[0,:,:4000,:]
    # compute fano factor
    fano_factor = (np.var(data, axis=0)/np.mean(data, axis=0)).flatten()
    nnans[i] = np.sum(np.isnan(fano_factor))
    # save mean over neurons and time windows without nans
    ff[i] = np.mean(fano_factor[np.isfinite(fano_factor)])
    print "ree = {} done".format(ree)
    print "ff = {}".format(ff[i])

np.save('ff_vs_ree', [ree_steps, ff, nnans])

plt.plot(ree_steps, ff)
plt.show()
