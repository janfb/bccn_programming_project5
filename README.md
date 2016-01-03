## Project 5: Slow Dynamics and High Variability in Networks with Clustered Connections

### Background
For long, network simulations have assumed homogeneous random connectivity profiles between neurons. Yet, many anatomical studies have demonstrated that especially connections between excitatory neurons are not uniformly distributed. Instead, one
can find clusters of highly connected neurons. 
Introducing clustering into network simulations can substantially change the behavior and dynamics of spiking neurons. Litwin-Kumar and Doiron (2012) have recently shown that even modest clustering of previously homogeneously connected networks can yield firing rate fluctuations and interesting attractor dynamics. In case of spontaneous activity, the network randomly samples different attractor states of cluster activations, Whereas an input stimulus can drive the network into a particular attractor and stabilize the corresponding attractor pattern.

### Problems
1.  Using BRIAN (www.briansimulator.org), implement a network of 5000 neurons (4000 excitatory and 1000 inhibitory ones) with homogeneous as well as heterogeneous clustered random connectivity between them.For various parameter values (e.g. time constants, synaptic strengths, etc.), you can refer to Litwin-Kumar and Doiron 2012. A description of the model can be
found in the supplementary material of the paper. You may only focus on the simulation of networks with non-overlapping clusters of equal size. What kind of network states do you observe in case of a homogeneously randomly connected network and a clustered one? Plot two exemplary raster plots for a qualitative comparison. 
2.  What measures can you apply to your simulations for a quantitative comparison of non-clustered and clustered networks? You might refer to statistics used in figure 2 of the paper, but you are free to use other measures as well.
3.  What crucial role does the parameter R_{EE} play? For which values do you observe changes  in  the  behavior  of  networks  compared  to  the  non-clustered  case?  To quantify changes in behavior you can consider the Fano Factor
as in Figure 4c of the paper. 
4. (optional) Study the evoked activity of your network model. To simulate evoked activity, you can slightly increase the constant input current mu of a subset of clusters (see Figure 7 of the paper). 
