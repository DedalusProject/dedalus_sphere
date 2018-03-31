from calculate_parameters_dynamo import calculate_parameters
import numpy as np

# resolutions are Nr x Nphi x Ntheta

# 24x24x24, alpha_BC=0, CNAB2 dt = 5e-6
KE_max_24_24_a0_CNAB2_5en6 = 37148.70
KE_min_24_24_a0_CNAB2_5en6 = 33454.07
ME_max_24_24_a0_CNAB2_5en6 = 1123.6242
ME_min_24_24_a0_CNAB2_5en6 = 1027.5675
data = np.loadtxt('data/marti_dynamo_E_24_24_a0_CNAB2_5en6.dat')
calculate_parameters(data[-20000:,0],data[-20000:,2])
calculate_parameters(data[-20000:,0],data[-20000:,3])

# 32x32x32, alpha_BC=0, CNAB2 dt = 5e-6
KE_max_32_32_a0_CNAB2_5en6 = 37435.33
KE_min_32_32_a0_CNAB2_5en6 = 33670.85
ME_max_32_32_a0_CNAB2_5en6 = 960.6942
ME_min_32_32_a0_CNAB2_5en6 = 882.8110
data = np.loadtxt('data/marti_dynamo_E_32_32_a0_CNAB2_5en6.dat')
calculate_parameters(data[-20000:,0],data[-20000:,2]/(3/2))
calculate_parameters(data[-20000:,0],data[-20000:,3]/(3/2))

# 48x48x48, alpha_BC=0, CNAB2 dt = 5e-6
KE_max_48_48_a0_CNAB2_5en6 = 37444.97
KE_min_48_48_a0_CNAB2_5en6 = 33681.17
ME_max_48_48_a0_CNAB2_5en6 = 943.8372
ME_min_48_48_a0_CNAB2_5en6 = 868.0984
data = np.loadtxt('data/marti_dynamo_E_48_48_a0_CNAB2_5en6.dat')
calculate_parameters(data[-20000:,0],data[-20000:,2])
calculate_parameters(data[-20000:,0],data[-20000:,3])

# 64x64x64, alpha_BC=2, SBDF4 dt = 2.5e-6
KE_max_64_64_a2_SBDF4_2p5en6 = 37444.32
KE_min_64_64_a2_SBDF4_2p5en6 = 33681.31
ME_max_64_64_a2_SBDF4_2p5en6 = 943.4111
ME_min_64_64_a2_SBDF4_2p5en6 = 867.7413
data = np.loadtxt('data/marti_dynamo_E_64_64_a2_SBDF4_2p5en6.dat')
calculate_parameters(data[-20000:,0],data[-20000:,1])
calculate_parameters(data[-20000:,0],data[-20000:,2])

# 128x64x64, alpha_BC=2, SBDF4 dt = 2.5e-6
KE_max_128_64_a2_SBDF4_2p5en6 = 37444.32
KE_min_128_64_a2_SBDF4_2p5en6 = 33681.31
ME_max_128_64_a2_SBDF4_2p5en6 = 943.4111
ME_min_128_64_a2_SBDF4_2p5en6 = 867.7413
data = np.loadtxt('data/marti_dynamo_E_128_64_a2_SBDF4_2p5en6.dat')
calculate_parameters(data[-20000:,0],data[-20000:,2])
calculate_parameters(data[-20000:,0],data[-20000:,3])

# 64x96x96, alpha_BC=2, SBDF4 dt = 2.5e-6
KE_max_64_96_a2_SBDF4_2p5en6 = 37444.24
KE_min_64_96_a2_SBDF4_2p5en6 = 33681.25
ME_max_64_96_a2_SBDF4_2p5en6 = 943.4046
ME_min_64_96_a2_SBDF4_2p5en6 = 867.7262
data = np.loadtxt('data/marti_dynamo_E_64_96_a2_SBDF4_2p5en6.dat')
calculate_parameters(data[-40000:,0],data[-40000:,2])
calculate_parameters(data[-40000:,0],data[-40000:,3])

# 64x128x128, alpha_BC=2, SBDF4 dt = 2.5e-6
KE_max_64_128_a2_SBDF4_2p5en6 = 37444.24
KE_min_64_128_a2_SBDF4_2p5en6 = 33681.26
ME_max_64_128_a2_SBDF4_2p5en6 = 943.4048
ME_min_64_128_a2_SBDF4_2p5en6 = 867.7265
data = np.loadtxt('data/marti_dynamo_E_64_128_a2_SBDF4_2p5en6.dat')
calculate_parameters(data[-20000:,0],data[-20000:,2])
calculate_parameters(data[-20000:,0],data[-20000:,3])

# 64x128x128, alpha_BC=2, SBDF4 dt = 1.25e-6
KE_max_64_128_a2_SBDF4_1p25en6 = 37444.24
KE_min_64_128_a2_SBDF4_1p25en6 = 33681.26
ME_max_64_128_a2_SBDF4_1p25en6 = 943.4048
ME_min_64_128_a2_SBDF4_1p25en6 = 867.7265
data = np.loadtxt('data/marti_dynamo_E_64_128_a2_SBDF4_1p25en6.dat')
calculate_parameters(data[-80000:,0],data[-80000:,2])
calculate_parameters(data[-80000:,0],data[-80000:,3])

