
# Alcune molecole utilizzate negli esempi
import numpy as np
from pyscf import gto, scf

# array delle distanze
arr_d = np.arange(0.3, 4, .05) # da 0.3 a 3.95 (Angstrom) con passo 0.05
# va specificata una base
basis = 'sto-6g'
# e una distanza interatomica
distanza = 1.

# idrogeno biatomico
H_2 = "Li .0 .0 .0; Li .0 .0 " + str(distanza)

mol = gto.M(
    atom=H_2,
    charge=0,
    spin=0,
    basis=basis,
    symmetry=True,
    verbose=0
)

# ione idrogenonio
altezza = np.sqrt(distanza**2 - (distanza/2)**2)

H_3_più = "H .0 .0 .0; H .0 .0 " + str(distanza) + "; H .0 " + str(altezza) + " " + str(distanza/2)

mol = gto.M(
    atom=H_2,
    charge=1,
    spin=0,
    basis=basis,
    symmetry=True,
    verbose=0
)

# litio biatomico
Li_2 = "Li .0 .0 .0; Li .0 .0 " + str(distanza)

mol = gto.M(
    atom=Li_2,
    charge=0,
    spin=0,
    basis=basis,
    symmetry=True,
    verbose=0
)

# ossido di litio (ossigeno al vertice in alto)
Li_2O = "Li .0 .0 .0; Li .0 .0 " + str(distanza) + "; O " + str(altezza) + " " + str(distanza/2)

mol = gto.M(
    atom=Li_2O,
    charge=0,
    spin=0,
    basis=basis,
    symmetry=True,
    verbose=0
)