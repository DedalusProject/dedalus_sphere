import numpy as np

def final_energy(file):
  data = np.loadtxt(file)
  return np.round(data[1,-1],decimals=8)

E_16_a0_S_8en5 = final_energy('data/marti_conv_E_16_16_a0_SBDF4_8en5.dat')
E_24_a0_S_8en5 = final_energy('data/marti_conv_E_24_24_a0_SBDF4_8en5.dat')
E_32_10_S_1en5 = final_energy('data/marti_conv_E_32_32_a0_SBDF4_1en5.dat')

E_32_a2_S_8en5 = final_energy('data/marti_conv_E_32_32_a2_SBDF4_8en5.dat')
E_32_a2_S_4en5 = final_energy('data/marti_conv_E_32_32_a2_SBDF4_4en5.dat')
E_32_a2_S_2en5 = final_energy('data/marti_conv_E_32_32_a2_SBDF4_2en5.dat')
E_32_a2_S_1en5 = final_energy('data/marti_conv_E_32_32_a2_SBDF4_1en5.dat')
E_32_a2_S_5en6 = 29.12045489 # I lost the raw data

E_32_a2_C_8en5 = final_energy('data/marti_conv_E_32_32_a2_CNAB2_8en5.dat')
E_32_a2_C_4en5 = final_energy('data/marti_conv_E_32_32_a2_CNAB2_4en5.dat')
E_32_a2_C_2en5 = final_energy('data/marti_conv_E_32_32_a2_CNAB2_2en5.dat')
E_32_a2_C_1en5 = final_energy('data/marti_conv_E_32_32_a2_CNAB2_1en5.dat')

E_64_a2_S_1en5 = 29.12045489 # I lost the raw data
E_64_a2_S_5en6 = 29.12045489 # I lost the raw data

