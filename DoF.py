
L_max = 63
#M_max = 8
M_max = L_max
N_max = 31
R_max = 3
dedalus = True

DoF = 0

for m in range(M_max+1):
  for ell in range(m,L_max+1):
    if dedalus:
      DoF += N_max + 1 - (ell-R_max)//2
    else:
      DoF += N_max

print(DoF)


