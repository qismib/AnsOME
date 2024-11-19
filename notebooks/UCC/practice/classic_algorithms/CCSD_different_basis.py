# Facciamola più semplice

import numpy as np
from pyscf import gto, scf, fci, mp, cc

# dichiaro l'array delle distanze
arr_d = np.arange(0.3, 4, .05) # da 0.3 a 3.95 (Angstrom) con passo 0.05

# vettore con le basi da provare
arr_basis = ['6-31g', 'sto-6g', 'cc-pvdz', 'aug-cc-pvdz']

# array per le energie
arr_energies = {
    '6-31g':       [],
    'sto-6g':      [],
    'cc-pvdz':     [],
    'aug-cc-pvdz': []
}

# un iterazione per ogni base
for basis in arr_basis:
    
    # una molecola per ogni distanza
    for dist in arr_d:
        geometry = "H .0 .0 .0; H .0 .0 " + str(dist)
        mol = gto.M(
            atom=geometry,
            charge=0,
            spin=0,
            basis=basis,
            symmetry=True,
            verbose=0
        )
        # per ciascuna molecola calcolo restricted HF 
        cm  = scf.RHF(mol) # campo medio
        e_HF = cm.kernel()  # chiamando il kernel otteniamo l'energia calcolata dal metodo
        
        # quindi uso cm per calcolare la correzione a e_HF data da CCSD 
        ccsd_calc = cc.CCSD(cm)
        e_ccsd  = ccsd_calc.kernel()[0]
        e_ccsd += e_HF # <- questa linea è necessaria perché CCSD generalmente mostra l'energia di differenza con HF
        arr_energies[basis].append(e_ccsd)
        

# plot

import matplotlib.pyplot as plt

for basis in arr_basis:
    plt.plot(arr_d, arr_energies[basis], label=basis)
    
# r opzione che ignora gli escape e permette di scrivere in LaTeX (\AA angstrom in Latex)
plt.title("Ground-State Energy w/ CSSD")
plt.xlabel(r"$d$ $[\AA]$")
plt.ylabel(r"Energy $[Ha]$")
plt.legend()
plt.show()
