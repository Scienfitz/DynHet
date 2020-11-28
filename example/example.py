from DynHet import calc_S, calc_F, read_lammps_disp
import numpy as np
import matplotlib.pyplot as plt

### Caclculate and plot S(r)
strFile = './250K-0.9466_ox.rdf' #file was created with VMD
bla     = np.genfromtxt(strFile)
r, g    = bla[:, 0], bla[:, 1]

q, S = calc_S(r, g, np.arange(0.01, 10.0, 0.001)) # calcualte for 1 from 0.01 to 10

plt.subplot(1, 3, 1)
plt.plot(r, g)
plt.gca().set_xlim((2, 7))
plt.title('g(r)')

plt.subplot(1, 3, 2)
plt.plot(q, S)
plt.title('S(q)')

plt.subplot(1, 3, 3)
plt.plot(q, S)
plt.gca().set_xlim((1.8, 1.86))
plt.gca().set_ylim((1.33, 1.42))
plt.title('S(q) zoom')

plt.gcf().set_size_inches((24, 6))

### Read lammps displacements
strFile = './npttest.displacements'

dyn = read_lammps_disp(strFile, 15000) #max of 15000 frames
dyn = dyn[:,0::3,:] # consider only oxygens which is every 3rd entry
print(f'Shape of the displacements var: {dyn.shape}')

### Calculate F, X
# frame interval is 0.1 ps
# as q we take the max of S(q) at around 1.84
# the max of the time axis is 1000 ps = 1 ns
# we consider only every 5th datapoint to reduce a bit teh comp cost
# we compute for around 400 datapoints on the log-spaced x axis (= time axis)
# we compute for the q value in the [1,1,1] direction
t, F, X = calc_F(dyn, dt=0.1, q=1.84, maxt=1000, nLagSteps=5, numSteps=400, nDirections=[1,1,1])

# this woudl do the same but average the result for 50 random directions
# t, F, X = calc_F(dyn, dt=0.1, q=1.84, maxt=1000, nLagSteps=5, numSteps=400, nDirections=50)

### Plot result
plt.subplot(2, 2, 1)
plt.semilogx(t, np.abs(F)) # F and X are complex values and we plot the real absolute values
plt.gca().set_ylim((0, 1))

plt.subplot(2, 2, 2)
plt.semilogx(t, np.abs(X))

plt.subplot(2, 2, 3)
plt.plot(t, np.abs(F))
plt.gca().set_ylim((0, 1))

plt.subplot(2, 2, 4)
plt.plot(t, np.abs(X))

plt.gcf().set_size_inches((20, 12))