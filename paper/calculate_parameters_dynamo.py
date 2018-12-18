import numpy as np

def calculate_parameters(t,energy):

  # find maxima
  max = 0*energy
  for i in range(len(max)-2):
    if energy[i+1] > energy[i+2] and energy[i+1] > energy[i]: max[i+1] = 1

  maxes = np.where(max>0)
  i_start = maxes[0][0]
  i_end = maxes[0][-1]

  t_new = t[i_start:i_end]
  energy_new = energy[i_start:i_end]

  energy_hat = np.fft.rfft(energy_new)

  A = np.abs(energy_hat[0])/len(energy_new)
  C = 2*np.max(np.abs(energy_hat[1:]))/len(energy_new)
  deltaT = (t[i_end]-t[i_start])/(np.sum(max)-1)
  freq = 1/deltaT

  print(A,C)

  return A,C,freq

