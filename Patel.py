import numpy as np
import math

# ======================================================================
def cnot_mtrx(ctrl, targ, nq):
  # targ is a row and ctrl is a column
  mtrx = np.identity(nq, dtype=int)
  mtrx[targ,ctrl] = 1
  return mtrx
# ======================================================================


# ======================================================================
def Lwr_CNOT_Synth(A, m):
  n = A.shape[0]

  CNOT_arr = []
  # iterate over column sections
  for sec in range( math.ceil(n/m) ):
    sz = m
    if (sec+1)*m > n:
      sz = n - sec*m      

    # remove duplicate sub-rows in section sec
    patt = [-1] * 2**sz # marker for first positions of sub-row patterns
    for row in range(sec*m,n):
      srp = A[row,sec*m:sec*m+sz]
      int_srp = sum([x*2**(sz-i-1) 
                     for i,x in enumerate(np.nditer(srp))])
      if int_srp == 0:
        continue
      # if first copy of patern save otherwise remove
      if patt[int_srp] == -1:
        patt[int_srp] = row
      else:
        A[row,:] += A[patt[int_srp],:]
        A %= 2
        # Step A
        CNOT_arr.append((patt[int_srp], row))
        #print("CNOT", patt[int_srp]+1, row+1)

    # use Gaussian elimination for remaining entries in column section
    for col in range(sec*m,sec*m+sz):
      # check for 1 on diagonal
      diag_one = A[col,col] == 1

      # remove ones in rows below column col
      for row in range(col+1,n):
        if A[row,col] == 1:
          if not diag_one:
            # Step B
            A[col,:] += A[row,:]
            A %= 2
            CNOT_arr.append((row, col))
            #print("CNOT", row+1, col+1)
            diag_one = True
          A[row,:] += A[col,:]
          A %= 2
          CNOT_arr.append((col, row))
          #print("CNOT", col+1, row+1)
  return A, CNOT_arr
# ======================================================================
