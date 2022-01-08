import math
import cmath
import numpy as np
import ChuLiuEdmonds
import Patel

# TOBEDONE: 
# - remove the same parity gadgets
# - add CNOTs to the parity gadgets

# ======================================================================
class zx_circ_cls():
  def __init__(self,nq):
    self.nq = nq
    self.op = []


  con_Y = {"I":"I", "X":"X", "Y":"Z", "Z":"Y"}
  con_Y_coef = {"I":1, "X":1, "Y":1, "Z":-1}


  con_X = {"I":"I", "X":"Z", "Y":"Y", "Z":"X"}
  con_X_coef = {"I":1, "X":1, "Y":-1, "Z":1}


  con_CX = {"II":"II", "XI":"XX", "YI":"YX", "ZI":"ZI",
            "IX":"IX", "XX":"XI", "YX":"YI", "ZX":"ZX",
            "IY":"ZY", "XY":"YZ", "YY":"XZ", "ZY":"IY",
            "IZ":"ZZ", "XZ":"YY", "YZ":"XY", "ZZ":"IZ"}
  con_CX_coef = {"II":1, "XI":1, "YI":1, "ZI":1,
                 "IX":1, "XX":1, "YX":1, "ZX":1,
                 "IY":1, "XY":1, "YY":-1, "ZY":1,
                 "IZ":1, "XZ":-1, "YZ":1, "ZZ":1}


  dct_XYZ_012 = {"I":-1, "X":0, "Y":1, "Z":2}
  dct_012_XYZ = {0:"X", 1:"Y", 2:"Z"}

  score_P = {"Z":0, "X":1, "Y":1.5}

  IXYZ = {
    "I": np.matrix([[1, 0],[0, 1]]),
    "X": np.matrix([[0, 1],[1, 0]]),
    "Y": np.matrix([[0,-1j],[1j, 0]],dtype=complex),
    "Z": np.matrix([[1, 0],[0,-1]]),
    }

  @staticmethod
  def is_pauli_gadget(op):
    return (len(op[0]) != 1) and (not "C" in op[0]) and (not "R" in op[0])


  @staticmethod
  def ps2pg(ps, nq, coef=1):
    q_arr = []
    for n,p in enumerate(ps):
      if p == "I":
        continue
      q_arr.append(nq-1-n)
    return (ps, q_arr, coef)


  @classmethod
  def conjugate_sqC_pg(cls, pg_op, q, sqC, nq):
    # Conjugate Pauli Gadget with single-qubit Cliffords (sqC)
    ps = pg_op[0]
    coef = 1

    if sqC == "X":
      coef = pg_op[2] * cls.con_X_coef[ps[nq-1-q]]
      ps = ps[:nq-1-q] + cls.con_X[ps[nq-1-q]] + ps[nq-q:]

    if sqC == "Y":
      coef = pg_op[2] * cls.con_Y_coef[ps[nq-1-q]]
      ps = ps[:nq-1-q] + cls.con_Y[ps[nq-1-q]] + ps[nq-q:]
    
    return (ps, cls.ps2pg(ps, nq)[1], coef)


  @classmethod
  def conjugate_cx_pg(cls, pg_op, ctrl, targ, nq):
    pg = pg_op[0]

    p = pg[nq-1-ctrl] + pg[nq-1-targ]
    p_ctrl, p_targ = cls.con_CX[p]

    if ctrl > targ:
      ps = pg[:nq-1-ctrl] + p_ctrl \
         + pg[nq-ctrl:nq-1-targ] + p_targ \
         + pg[nq-targ:]
    else:
      ps = pg[:nq-1-targ] + p_targ \
         + pg[nq-targ:nq-1-ctrl] + p_ctrl \
         + pg[nq-ctrl:]

    coef = pg_op[2]*cls.con_CX_coef[p]

    return cls.ps2pg(ps, nq, coef)


  @staticmethod
  def CX_mtrx(ctrl, targ, nq):
    mtrx = np.identity((2**nq))

    for r in range(2**nq):
      r_bin = bin(r)[2:].zfill(nq)

      if int(r_bin[nq-1-ctrl]) == 1:
        mtrx[r,r] = 0

        c_bin = r_bin[:nq-1-targ] + str((int(r_bin[nq-1-targ])+1)%2) + r_bin[nq-targ:]
        c = int(c_bin, 2)

        mtrx[r,c] = 1
        mtrx[c,r] = 1

    return mtrx


  @staticmethod
  def H_mtrx(q, nq):
    mtrx = np.kron( np.identity((2**(nq-1-q))), 
                    np.kron( 
                     np.matrix([[1,1],[1,-1]])/math.sqrt(2),
                     np.identity((2**q))
                     )
                   )
    return mtrx


  @staticmethod
  def m_mtrx(q, nq):
    mtrx = np.kron( np.identity((2**(nq-1-q))), 
                    np.kron( 
                     np.matrix([[1,1j],[1j,1]])/math.sqrt(2),
                     np.identity((2**q))
                     )
                   )
    return mtrx


  @staticmethod
  def p_mtrx(q, nq):
    mtrx = np.kron( np.identity((2**(nq-1-q))), 
                    np.kron( 
                     np.matrix([[1,-1j],[-1j,1]])/math.sqrt(2),
                     np.identity((2**q))
                     )
                   )
    return mtrx


  @staticmethod
  def PG_mtrx(op):
    nq = len(op[0])
    
    for i,p in enumerate(op[0]):
      if i == 0:
        mtrx = zx_circ_cls.IXYZ[p]
      else:
        mtrx = np.kron(mtrx, zx_circ_cls.IXYZ[p])

    if op[0].count("X") == 0 and op[0].count("Y") == 0:
      res = np.identity(2**nq, dtype=complex)
      for i in range(2**nq):
        res[i,i] = cmath.exp(1j * op[2] * mtrx[i,i])
    else:
      eigen_val, eigen_vec = np.linalg.eig(mtrx)
      res = np.zeros((2**nq,2**nq))
      for row in range(2**nq):
        P = np.outer(eigen_vec[:,row], np.conj(eigen_vec[:,row]) )
        res = np.add(res,cmath.exp(1j * op[2] * eigen_val[row]) * P)

    return res
    
        
  def add_pauli_gadget(self, ps, coef=1):
    self.op.append( self.ps2pg(ps, self.nq, coef) )


  def first_pauli_gadget(self):
    first = -1
    for i,op in enumerate(self.op):
      if self.is_pauli_gadget(op):
        first = i
        break
    return first


  def last_pauli_gadget(self):
    last = len(self.op)
    for i,op in reversed(list(enumerate(self.op))):
      if self.is_pauli_gadget(op):
        last = i
        break
    return last


  def diag(self):
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()

    q_diag = [True]*self.nq
    for i in range(first,last+1):
      for q,p in enumerate(self.op[i][0]):
        if p == "X" or p == "Y":
          q_diag[self.nq-1-q] = False
    return q_diag


  def triv_diag(self):
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()
    
    q_diag = self.diag()

    diagonalized = False
    for q in range(self.nq):
      if q_diag[q]:
        continue

      x_diag = True
      y_diag = True
      for i in range(first,last+1):
        if self.op[i][0][self.nq-1-q] != "X" and \
           self.op[i][0][self.nq-1-q] != "I":
          x_diag = False
        if self.op[i][0][self.nq-1-q] != "Y" and \
           self.op[i][0][self.nq-1-q] != "I":
          y_diag = False
        if not x_diag and not y_diag:
          break
      
      if x_diag:
        self.conjugate(q, "X")
        diagonalized = True
        first = self.first_pauli_gadget()
        last = self.last_pauli_gadget()
      if y_diag:
        self.conjugate(q, "Y")
        diagonalized = True
        first = self.first_pauli_gadget()
        last = self.last_pauli_gadget()

    return diagonalized


  def compatible(self):
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()

    ps_arr = []
    for i in range(first,last+1):
      ps_arr.append( self.op[i][0] )


    compatible_pairs = []
    AB = []
    for q1 in range(self.nq-1):
      for q2 in range(q1+1,self.nq):

        AB_mtrx = np.matrix([[False,False,False],
                             [False,False,False],
                             [False,False,False]])
        AB_filled = np.matrix([[False,False,False],
                               [False,False,False],
                               [False,False,False]])

        for ps in ps_arr:
          if ps[q1] == "I" and ps[q2] == "I":
            continue

          if ps[q1] != "I" and ps[q2] != "I":
            mask = [ii for ii in range(3) if ii != self.dct_XYZ_012[ps[q2]]]
            AB_mtrx[ self.dct_XYZ_012[ps[q1]], mask ] = False
            AB_filled[ self.dct_XYZ_012[ps[q1]], mask ] = True

            mask = [ii for ii in range(3) if ii != self.dct_XYZ_012[ps[q1]]]
            AB_mtrx[ mask, self.dct_XYZ_012[ps[q2]] ] = False
            AB_filled[ mask, self.dct_XYZ_012[ps[q2]] ] = True

            if not AB_filled[ self.dct_XYZ_012[ps[q1]], self.dct_XYZ_012[ps[q2]] ]:
              AB_mtrx[ self.dct_XYZ_012[ps[q1]], self.dct_XYZ_012[ps[q2]] ] = True
              AB_filled[ self.dct_XYZ_012[ps[q1]], self.dct_XYZ_012[ps[q2]] ] = True

            if not AB_mtrx.any() and AB_filled.all():
              break

          if ps[q1] == "I":
            mask = [ii for ii in range(3) if ii != self.dct_XYZ_012[ps[q2]]]
            AB_mtrx[ :, mask ] = False
            AB_filled[ :, mask ] = True
            for ii in range(3):
              if not AB_filled[ ii, self.dct_XYZ_012[ps[q2]] ]:
                AB_mtrx[ ii, self.dct_XYZ_012[ps[q2]] ] = True
                AB_filled[ ii, self.dct_XYZ_012[ps[q2]] ] = True
            continue

          if ps[q2] == "I":
            mask = [ii for ii in range(3) if ii != self.dct_XYZ_012[ps[q1]]]
            AB_mtrx[ mask, : ] = False
            AB_filled[ mask, : ] = True
            for ii in range(3):
              if not AB_filled[ self.dct_XYZ_012[ps[q1]],ii ]:
                AB_mtrx[ self.dct_XYZ_012[ps[q1]], ii ] = True
                AB_filled[ self.dct_XYZ_012[ps[q1]], ii ] = True
            continue

        if AB_mtrx.any():
          compatible_pairs.append( (q1,q2) )
          tmp = []
          for r in range(3):
            for c in range(3):
              if AB_mtrx[r,c]:
                tmp.append( (self.dct_012_XYZ[r],self.dct_012_XYZ[c]) )
          AB.append(tmp)
    
    return compatible_pairs, AB


  def conjugate(self, q, sqC):
    # Conjugate with single-qubit Cliffords (sqC)
    # Find the first and last Pauli Gadgets
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()

    # Conjugate with single-qubit Cliffords
    ops = []
    for i in range(first,last+1):
      if not self.is_pauli_gadget(self.op[i]):
        print("Additional commutation is needed, which is not yet implemented!")
        exit()

      ps = self.op[i][0]
      coef = self.op[i][2]
      if sqC == "X":
        ps = self.op[i][0][:self.nq-1-q] \
             + self.con_X[self.op[i][0][self.nq-1-q]] \
             + self.op[i][0][self.nq-q:]
        coef = self.op[i][2]*self.con_X_coef[self.op[i][0][self.nq-1-q]]

      if sqC == "Y":
        ps = self.op[i][0][:self.nq-1-q] \
             + self.con_Y[self.op[i][0][self.nq-1-q]] \
             + self.op[i][0][self.nq-q:]
        coef = self.op[i][2]*self.con_Y_coef[self.op[i][0][self.nq-1-q]]

      ops += [(ps, self.op[i][1], coef)]

    ops_first = self.op[:first]
    ops_last = self.op[last+1:]
    if sqC == "X":
      ops_first += [("H",[q],1)]
      ops_last = [("H",[q],1)] + ops_last

    if sqC == "Y":
      ops_first += [("p",[q],1)]
      ops_last = [("m",[q],1)] + ops_last

    self.op = ops_first + ops + ops_last


  def conjugate_cx(self, ctrl, targ):
    #Conjugate with CX gate to diagonalise the target qubit
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()
    
    # Conjugate with CX gate to diagonalise the target qubit
    ops = []
    for i in range(first,last+1):
      if not self.is_pauli_gadget(self.op[i]):
        print("Additional commutation is needed, which is not yet implemented!")
        exit()

      p = self.op[i][0][self.nq-1-ctrl] + self.op[i][0][self.nq-1-targ]
      p_ctrl, p_targ = self.con_CX[p]

      if ctrl > targ:
        ps = self.op[i][0][:self.nq-1-ctrl] \
           + p_ctrl \
           + self.op[i][0][self.nq-ctrl:self.nq-1-targ] \
           + p_targ \
           + self.op[i][0][self.nq-targ:]
      else:
        ps = self.op[i][0][:self.nq-1-targ] \
           + p_targ \
           + self.op[i][0][self.nq-targ:self.nq-1-ctrl] \
           + p_ctrl \
           + self.op[i][0][self.nq-ctrl:]

      coef = self.op[i][2]*self.con_CX_coef[p]

      ops += [zx_circ_cls.ps2pg(ps,self.nq,coef)]

    ops_first = self.op[:first] + [("CX",[ctrl,targ],1)]
    ops_last = [("CX",[ctrl,targ],1)] + self.op[last+1:]

    self.op = ops_first + ops + ops_last


  def greedy_diag(self):
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()
    
    #print("first = ", first, "last = ", last)

    arr_sqC_l = []
    arr_sqC_r = []
    for i,q in enumerate(self.op[first][1]):
      if self.op[first][0][self.nq-1-q] == "X":
        arr_sqC_l.append( ("H",[q],1) )
        arr_sqC_r.append( ("H",[q],1) )
      if self.op[first][0][self.nq-1-q] == "Y":
        arr_sqC_l.append( ("p",[q],1) )
        arr_sqC_r.append( ("m",[q],1) )
    #print("sqC: ", arr_sqC_l)

    arr_cx_l = []
    for i in range(len(self.op[first][1])-1,0,-1):
      ctrl = self.op[first][1][i]
      targ = self.op[first][1][i-1]
      arr_cx_l.append( ("CX",[ctrl,targ],1) )
    #print("CX:", arr_cx_l)

    arr_cx_r = []
    for op in reversed(arr_cx_l):
      arr_cx_r.append(op)

    # commutating
    pg_arr = []
    for i in range(first+1,last+1):
      pg_arr.append( self.op[i] )
      #print(pg_arr[-1])

    # conjugate with single-qubit Cliffords
    for op in arr_sqC_r:
      #print("\n",op)
      for i in range(len(pg_arr)):
        #print("\n",pg_arr[i])
        q = op[1][0]
        sqC = self.op[first][0][self.nq-1-q]
        pg_arr[i] = zx_circ_cls.conjugate_sqC_pg(pg_arr[i], q, sqC, self.nq)
        #print(pg_arr[i])

    # conjugate with CX
    for op in arr_cx_l:
      #print("\n",op,"\n")

      for i in range(len(pg_arr)):
        #print( pg_arr[i], zx_circ_cls.conjugate_cx_pg(pg_arr[i],op[1][0],op[1][1],self.nq) )
        pg_arr[i] = zx_circ_cls.conjugate_cx_pg(pg_arr[i],op[1][0],op[1][1],self.nq)

    
    q_arr = [self.op[first][1][0]]
    p_str = "".join(["Z" if q == self.nq-1-q_arr[0] 
                     else "I" for q in range(self.nq)])
    coef = self.op[first][2]

    self.op = self.op[:first] \
      + arr_sqC_l + arr_cx_l + [(p_str, q_arr, coef)] \
          + pg_arr + arr_cx_r + arr_sqC_r + self.op[last+1:]


  def best_compatible(self, q_diag):
    arr_q, arr_AB = self.compatible()

    score_bst = self.nq*4
    qA_bst = -1
    qB_bst = -1
    A_bst = "I"
    B_bst = "I"
    for i,qAB in enumerate( arr_q ):
      qA = self.nq - 1 - qAB[0]
      qB = self.nq - 1 - qAB[1]

      # If both qubits are already diagonal: continue
      if q_diag[qA] and q_diag[qB]:
        continue

      add = 4 * ( abs(qA - qB) - 1 )

      for AB in arr_AB[i]:
        A, B = AB
        score = self.score_P[A] + self.score_P[B] + add
        if score < score_bst:
          score_bst = score
          qA_bst = qA
          qB_bst = qB
          A_bst = A
          B_bst = B

    if qA_bst == -1:
      return qA_bst, qB_bst, qA_bst, qB_bst

    if not q_diag[qA_bst] and not q_diag[qB_bst]:
      if qA_bst < qB_bst:
        q_ctrl = qA_bst
        P_ctrl = A_bst
        q_targ = qB_bst
        P_targ = B_bst
      else:
        q_ctrl = qB_bst
        P_ctrl = B_bst
        q_targ = qA_bst
        P_targ = A_bst
      return q_ctrl, q_targ, P_ctrl, P_targ

    if q_diag[qA_bst]:
      q_ctrl = qA_bst
      P_ctrl = A_bst
      q_targ = qB_bst
      P_targ = B_bst
    else:
      q_ctrl = qB_bst
      P_ctrl = B_bst
      q_targ = qA_bst
      P_targ = A_bst
    return q_ctrl, q_targ, P_ctrl, P_targ


  def cowtan(self):
    while True:
      q_diag = self.diag()
      #print(q_diag)
      if all(q_diag):
        break
      
      # Check whether there is a trivially diagonalisable qubit
      triv_diag = self.triv_diag()
      if triv_diag:
        #print("\nTrivial", "\n",self)
        #return #remove
        continue

      # Find compatible pair
      q_ctrl, q_targ, A, B = self.best_compatible(q_diag)

      # Perform diagonalization
      if q_ctrl != -1:
        self.conjugate(q_ctrl, A)
        self.conjugate(q_targ, B)
        self.conjugate_cx(q_ctrl, q_targ)
        #print("\nCompatible", q_ctrl, q_targ, A, B, "\n",self)
      # If none of the qubits can be diagonalized: greedy diagonalization
      else:
        #print("\nGreedy")
        self.greedy_diag()
        #print(self)
        #return #remove


  def parity_table(self):
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()
    
    table = np.zeros((self.nq, last-first+1), dtype=int)
    angles = np.zeros((last-first+1))
    for i in range(first,last+1):
      for q in self.op[i][1]:
        table[q,i-first] = 1
      angles[i-first] = self.op[i][2]
    return table, angles


  def vandaele_patel(self):
    # Vandaele part
    zx_oper_arr = []

    p_table, angles = self.parity_table()

    n_columns = p_table.shape[1]
    n_rows = self.nq
    #print("n_columns = ", n_columns, "n_rows = ", n_rows, p_table.shape)
    #print(p_table)
    #print("angles = ", angles)

    A = np.identity(n_rows, dtype=int)
    while True:
      # check if there are qubits with required parity
      for ic in range(n_columns):
        if sum(p_table[:,ic]) == 1:
          for ir,x in enumerate(p_table[:,ic]):
            if x == 1:
              p_str = "".join(["Z" if q == self.nq-1-ir
                               else "I" for q in range(self.nq)])
              zx_oper_arr.append( (p_str, [ir], angles[ic]) )
              p_table[ir,ic] = 0
              break

      # Find best y
      int_y_bst = int("1"*n_rows,2) + 1
      h_bst = n_columns + 1
      for i in range(n_columns):
        h = sum(p_table[:,i])
        if h <= 1:
          continue

        bn_y = "".join( [str(x) for x in reversed(p_table[:,i])] )
        int_y = int(bn_y,2)
        if h <= h_bst and int_y < int_y_bst:
          h_bst = h
          int_y_bst = int_y
          i_bst = i
          #print("\nBest = ", i_bst, "".join( [str(x) for x in reversed(p_table[:,i_bst])]))

      # create graph
      q_arr = []
      h_arr = []
      for i,x in enumerate(p_table[:,i_bst]):
        if x != 1:
          continue
        q_arr.append( i )
        h_arr.append( sum(p_table[i,:]) )

      nv = len(q_arr)
      weights_mtrx = np.zeros((nv,nv))

      for i in range( nv-1 ):
        for j in range( i+1, nv ):
          h_ij = sum([ p_table[q_arr[i],ic]^p_table[q_arr[j],ic] 
                      for ic in range(n_columns)])
          weights_mtrx[i,j] = h_ij - h_arr[j]
          weights_mtrx[j,i] = h_ij - h_arr[i]

      arborescence = ChuLiuEdmonds.find_MWSArborescence(weights_mtrx)
      #test_arborescence = ChuLiuEdmonds.straightforward_test(weights_mtrx, arborescence)
      #print(arborescence, test_arborescence)

      # Convert arborescence to CNOT gates
      for e in reversed(arborescence):
        ctrl = q_arr[e[1]]
        targ = q_arr[e[0]]
        zx_oper_arr.append( ("CX",[ctrl,targ],1) )

        A = Patel.cnot_mtrx(ctrl, targ, self.nq).dot(A)%2
        for ic in range(n_columns):
          p_table[ctrl,ic] = p_table[targ,ic]^p_table[ctrl,ic]

      root = arborescence[0][0]
      p_table[q_arr[root],i_bst] = 0

      p_str = "".join(["Z" if q == self.nq-1-q_arr[root]
                       else "I" for q in range(self.nq)])
      zx_oper_arr.append( (p_str, [q_arr[root]], angles[i_bst]) )
      
      if not p_table.any():
        break
      
    # Patel part
    A, cx_arr = Patel.Lwr_CNOT_Synth(A, 2)
    A, cx_arr_t = Patel.Lwr_CNOT_Synth(A.T, 2)

    for x in cx_arr:
      zx_oper_arr.append( ("CX",[x[0], x[1]],1) )
    for x in reversed(cx_arr_t):
      zx_oper_arr.append( ("CX",[x[1], x[0]],1) )
      
    # Remove Pauli Gadgets by Vandaele Patel circuit
    first = self.first_pauli_gadget()
    last = self.last_pauli_gadget()

    self.op = self.op[:first] + zx_oper_arr + self.op[last+1:]

  
  def to_matrix(self):
    mtrx = np.identity(2**self.nq, dtype=complex)
    
    for op in self.op:
      if op[0] == "H":
        mtrx = self.H_mtrx(op[1][0], self.nq).dot(mtrx)
        continue
      if op[0] == "p":
        mtrx = self.p_mtrx(op[1][0], self.nq).dot(mtrx)
        continue
      if op[0] == "m":
        mtrx = self.m_mtrx(op[1][0], self.nq).dot(mtrx)
        continue
      if op[0] == "CX":
        mtrx = self.CX_mtrx(op[1][0], op[1][1], self.nq).dot(mtrx)
        continue

      mtrx = self.PG_mtrx(op).dot(mtrx)
    return mtrx


  def stats(self):
    ncx = 0
    nsqC = 0
    for op in self.op:
      if op[0] == "CX":
        ncx += 1
      else:
        nsqC += 1
    return nsqC, ncx

  def qasm(self):
    lines = [""]*self.nq
    for op in self.op:
      #print(op)
      gate = ["\qw"]*self.nq

      if op[0] == "H":
        gate[op[1][0]] = "\gate{H}"
      if op[0] == "p":
        gate[op[1][0]] = "\gate{V}"
      if op[0] == "m":
        gate[op[1][0]] = "\gate{V^\dagger}"
      if op[0] == "CX":
        dist = op[1][1] - op[1][0]
        gate[op[1][0]] = "\ctrl{" + str(dist) + "}"
        gate[op[1][1]] = "\\targ{}"
      if op[0].count("Z") == 1:
        gate[op[1][0]] = "\gate{R_z("+str(op[2])+")}"
        #print(gate[op[1][0]])
        #exit()

      for i in range(self.nq):
        lines[i] += gate[i] + " & "

    for i,line in enumerate(lines):
      print(line)
      if i != self.nq-1:
        print("\\\\")


  def __str__(self):
    res = ""
    for op in self.op:
      if op[0] == "H":
        res += "H" + "_" + str(op[1][0]) + " "
        continue
      if op[0] == "p":
        res += "p" + "_" + str(op[1][0]) + " "
        continue
      if op[0] == "m":
        res += "m" + "_" + str(op[1][0]) + " "
        continue
      if op[0] == "CX":
        q_str = "".join([str(i) for i in op[1]])
        res += "CX" + "_" + q_str + " "
        continue

      q_str = "".join([str(i) for i in op[1]])
      p_str = "".join([p for p in op[0] if p != "I"])
      res += str(op[2]) + "*" + p_str + "_" + q_str + " "
    return res
# ======================================================================    
