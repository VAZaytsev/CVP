import Cowtan
import test_mod
import numpy as np
import sys

fl_inp = open(sys.argv[1],"r")

ps_arr = []
coef_arr = []
for ln in fl_inp.readlines():
  if ln == "\n":
    continue
  p,c = ln.split()
  ps_arr.append(p)
  coef_arr.append(float(c))

# Test whether all given Pauli strings commute
for i in range(len(ps_arr)-1):
  for j in range(i,len(ps_arr)):
    if not test_mod.ps_commute(ps_arr[i],ps_arr[j]):
      print(ps_arr[i],"and",ps_arr[j],"do not commute")
      exit()

mtrx = test_mod.mtrx_from_input(ps_arr, coef_arr)

Nq = len(ps_arr[0])
print("Nq = ", Nq)

# Create zx circuit
zx_circ = Cowtan.zx_circ_cls(Nq)
for i,ps in enumerate(ps_arr):
  zx_circ.add_pauli_gadget(ps, coef=coef_arr[i])
#print(zx_circ)
diff = np.amax(abs(zx_circ.to_matrix() - mtrx))
if diff > 1.e-13:
  print("Something wrong! Abort!")
  exit()


zx_circ.cowtan()
print("\nAfter Cowtan")
print(zx_circ)
diff = np.amax(abs(zx_circ.to_matrix() - mtrx))
if diff > 1.e-13:
  print("Something wrong with Cowtan! Abort!")
  exit()


zx_circ.vandaele_patel()
print("\nAfter Cowtan, Vandaele, and Patel")
print(zx_circ)
diff = np.amax(abs(zx_circ.to_matrix() - mtrx))
if diff > 1.e-13:
  print("Something wrong with Vandaele and Patel! Abort!")
  exit()

nsqC, ncx = zx_circ.stats()
print( "\nSingle Qubit Gates = ", nsqC, "CNOTs = ", ncx )

# To print circuit for plot in qasm
#zx_circ.qasm()

#print( zx_circ.to_matrix() )
