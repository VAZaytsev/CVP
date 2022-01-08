Simplification of the circuits consisting of commuting Pauli gadgets. The algorithm consists of several steps
1. Simultaneous diagonalization of the Pauli strings, which form Pauli gadgets, by conjugating with the single-qubit Clifford and CNOT gates as described by Cowtan, Simmons and Duncan. 
2. Representation of the phase gadgets, obtained on the previous step, by the Z rotations and CNOT gates in accordance to work by Vandaele, Martiel and de Brugiere.
3. Restore the parity of the qubits, being changed on the previous step, by the algorithm proposed by Patel, Markov, and Hayes.

# References 

A. Cowtan, W. Simmons, and R. Duncan,
''A Generic Compilation Strategy for the Unitary Coupled Cluster Ansatz'',
[arXiv:2007.10515](https://arxiv.org/abs/2007.10515)

V. Vandaele, S. Martiel, and T.G. de Brugiere,
''Phase polynomials synthesis algorithms for NISQ architectures and beyond'',
[arXiv:2104.00934](https://arxiv.org/abs/2104.00934)

K.N. Patel, I.L. Markov, and J.P.Hayes,
''Optimal synthesis of linear reversible circuits'',
[Quantum Inf. and Comp. 8, 0282 (2008)](http://dx.doi.org/10.26421/QIC8.3-4-4)
