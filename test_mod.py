import numpy as np
import cmath

IXYZ = {
  "I": np.matrix([[1, 0],[0, 1]]),
  "X": np.matrix([[0, 1],[1, 0]]),
  "Y": np.matrix([[0,-1j],[1j, 0]],dtype=complex),
  "Z": np.matrix([[1, 0],[0,-1]])
  }

# ======================================================================
def exp_alpha_PS(alpha, ps):
  nq = len(ps)
  for i in range(nq):
    if i == 0:
      mtrx_pwr = IXYZ[ps[i]]
    else:
      mtrx_pwr = np.kron(mtrx_pwr, IXYZ[ps[i]])

  eigen_val, eigen_vec = np.linalg.eig(mtrx_pwr)

  mtrx_out = np.zeros((2**nq,2**nq))
  for row in range(2**nq):
    P = np.outer(eigen_vec[:,row], np.conj(eigen_vec[:,row]) )
    mtrx_out = np.add(mtrx_out,cmath.exp(alpha * eigen_val[row]) * P)

  return mtrx_out
# ======================================================================


# ======================================================================
def mtrx_from_input(ps_arr, coef_arr):
  nq = len(ps_arr[0])
  mtrx = np.identity(2**nq, dtype=complex)

  for i, ps in enumerate(ps_arr):
    ps_mtrx = exp_alpha_PS(1j*coef_arr[i], ps)
    mtrx = ps_mtrx.dot(mtrx)

  return mtrx
# ======================================================================


# ======================================================================
def ps_commute(ps1,ps2):
  coef = 1
  for i in range(len(ps1)):
    if ps1[i] == "I":
      continue
    if ps2[i] == "I":
      continue
    if ps1[i] == ps2[i]:
      continue
    coef = -coef

  return coef == 1
# ======================================================================
