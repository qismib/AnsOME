
# Integrali a uno e due elettroni con PySCF

from pyscf import gto

# Molecola idruro di litio
LiH = "Li .0 .0 .0; H .0 .0 0.600"
mol = gto.M(atom=LiH, basis='sto-6g')
two_electron_integrals = mol.intor('int2e') # , aosym='s8') opzioni di simmetria
one_electron_integrals = mol.intor("int1e_kin") + mol.intor("int1e_nuc")


print ("ONE-ELECTRON")
print(one_electron_integrals)

print ("TWO-ELECTRON")
print(two_electron_integrals)