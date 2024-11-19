import numpy as np
from scipy.linalg import expm
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers   import PySCFDriver
from typing import Tuple, Sequence, Union, List
from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
from qiskit_nature.second_q.problems  import ElectronicBasis
from time import time
from pyscf import gto, scf, fci, ao2mo

from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper



''' Problem '''
def generate_problem (molecule, basis: str = 'sto3g'):
    
    driver = PySCFDriver(
        atom=molecule,
        basis=basis, # 3 gaussiane per approssimare una funzione di Slater
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    mo_problem = driver.run()
    
    return mo_problem

''' Ansatz '''
from qiskit_nature.second_q.circuit.library import HartreeFock, PUCCD

def generate_puccd (mo_problem, mapper: QubitMapper = JordanWignerMapper()):

    puccd = PUCCD(
        mo_problem.num_spatial_orbitals,
        mo_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            mo_problem.num_spatial_orbitals,
            mo_problem.num_particles,
            mapper,
        ),
    )
    
    return puccd


from qiskit_algorithms import VQE, VQEResult
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator # Estimator deprecated 

''' Compute puccd optimal point & assign optimal parameters'''
def compute_puccd_optimal (puccd, 
                           problem, 
                           mapper: QubitMapper = JordanWignerMapper(), 
                           ini: np.ndarray = None, 
                           opt = SLSQP()
                           ):
    
    ops = problem.second_q_ops()
    main_op = mapper.map(ops[0])
    aux_ops = mapper.map(ops[1])
    
    # set default bounds
    puccd._bounds = [[-np.pi,np.pi] for _ in range(puccd.num_parameters)]

    ini_solver = VQE(Estimator(), puccd, opt)
    
    if ini is not None:
        ini_solver.initial_point=ini
        
    ini_result = ini_solver.compute_minimum_eigenvalue(operator=main_op, aux_operators=aux_ops)
    
    return ini_result



def create_orbital_rotation_list(ansatz) -> list:
    """ Creates a list of indices of matrix kappa that denote the pairs of orbitals that
    will be rotated. For instance, a list of pairs of orbital such as [[0,1], [0,2]]. """
    
    half_as = int(ansatz.num_qubits / 2)
    
    orbital_rotations = []
    
    # TODO: 
    '''
    # list is built according to frozen orbitals
    if self._frozen_list:
        for i in range(half_as):
            if i in self._frozen_list:
                continue
            for j in range(half_as):
                if j in self._frozen_list:
                    continue
                if i < j:
                    self._orbital_rotations.append([i, j])
    else:
        for i in range(half_as):
            for j in range(half_as):
                if i < j:
                    self._orbital_rotations.append([i, j])
    '''
    
    for i in range(half_as):
            for j in range(half_as):
                if i < j:
                    orbital_rotations.append([i, j])
                    
    return orbital_rotations

''' cambiare il formato della lista '''
def convert_rotations_list (rotations):
    excitations = [([i], [j]) for i, j in rotations]
    return excitations


def build_fermionic_excitation_ops(
    mo_problem, 
    excitations: Sequence,
    mapper: QubitMapper = JordanWignerMapper()
    ) -> list[FermionicOp]:
    """Builds all possible excitation operators with the given number of excitations for the
    specified number of particles distributed in the number of orbitals.

    Args:
        excitations: the list of excitations.

    Returns:
        The list of excitation operators in the second quantized formalism.
    """
    num_spin_orbitals = 2 * mo_problem.num_spatial_orbitals
    operators = []

    for exc in excitations:
        label = []
        for occ in exc[0]:
            label.append(f"+_{occ}")
        for unocc in exc[1]:
            label.append(f"-_{unocc}")
        op = FermionicOp({" ".join(label): 1}, num_spin_orbitals=num_spin_orbitals)
        op_adj = op.adjoint()
        # we need to account for an additional imaginary phase in the exponent accumulated from
        # the first-order trotterization routine implemented in Qiskit
        op_minus = 1j * (op - op_adj)
        operators.append(op_minus)
        
        # map to qubit operators
        qubit_ops = mapper.map(operators)

    return qubit_ops


from qiskit.circuit.library import EvolvedOperatorAnsatz

def generate_oo_puccd (puccd, mo_problem, excitations):

    qubit_ops = build_fermionic_excitation_ops(mo_problem, excitations)
    
    oo_puccd = EvolvedOperatorAnsatz(
        operators=qubit_ops,
        initial_state=puccd,
        parameter_prefix='k'
    )
    
    return oo_puccd


def orbital_rotation_matrix(problem, parameters: np.ndarray, rotations: list) -> Tuple[np.ndarray, np.ndarray]:
    """ Creates 2 matrices K_alpha, K_beta that rotate the orbitals through MO coefficient
    C_alpha = C_RHF * U_alpha where U = e^(K_alpha), similarly for beta orbitals. """

    dim_kappa_matrix = problem.num_spatial_orbitals
    
    k_matrix_alpha = np.zeros((dim_kappa_matrix, dim_kappa_matrix))
    k_matrix_beta  = np.zeros((dim_kappa_matrix, dim_kappa_matrix))
    
    for i, exc in enumerate(rotations):
        k_matrix_alpha[exc[0]][exc[1]] =  parameters[i]
        k_matrix_alpha[exc[1]][exc[0]] = -parameters[i]
        k_matrix_beta[exc[0]][exc[1]]  =  parameters[i]
        k_matrix_beta[exc[1]][exc[0]]  = -parameters[i]

    matrix_a = expm(k_matrix_alpha)
    matrix_b = expm(k_matrix_beta)
    
    return matrix_a, matrix_b


from qiskit_nature.second_q.operators import ElectronicIntegrals

def rotate_orbitals(problem, one_body, two_body, matrix_a, matrix_b, mapper = JordanWignerMapper()): 
    """Doctstring"""
    
    # Matrice C (6x6) e integrali a un corpo (6x6)
    C = matrix_a  
    h = one_body 
    eri = two_body
    
    N = len(C)

    # Trasformazione degli integrali a un corpo
    h_transformed = np.einsum('up,uv,vq->pq', C, h, C)

    # Trasformazione degli integrali a due corpi
    eri_transformed = np.einsum('up,vq,xr,ys,uvxy->pqrs', C, C, C, C, eri)
    eri_transformed = np.zeros((N, N, N, N))
    
    # definisco un oggetto ElectronicIntegrals per modificare quello contenuto in mo_problem
    e_int = ElectronicIntegrals.from_raw_integrals(h_transformed, eri_transformed)

    problem.hamiltonian.ElectronicIntegrals = e_int
    
    # extract rotated operator
    rotated_operator = problem.hamiltonian.second_q_op()
    rotated_operator = mapper.map(rotated_operator)
    
    return rotated_operator


def energy_evaluation_oo(
    problem,
    one_body,
    two_body,
    solver, 
    rotations,
    num_parameters_puccd: int,
    callback: list,
    parameters: np.ndarray,
) -> Union[float, List[float]]:
    """Doctstring"""
    
    # splicing
    puccd_parameter_values    = parameters[:num_parameters_puccd ] 
    rotation_parameter_values = parameters[ num_parameters_puccd:] 

    # CALCULATE COEFFICIENTS OF ROTATION MATRIX HERE:
    matrix_a, matrix_b = orbital_rotation_matrix(problem, rotation_parameter_values, rotations)
    
    #print('pUCCD parameters:\n', puccd_parameter_values)
    #print('Rotation parameters:\n', rotation_parameter_values)
    #print("Nature matrix a:\n", matrix_a)
    #print("Nature matrix b:\n", matrix_b)
    
    # ROTATE AND RECOMPUTE OPERATOR HERE:
    rotated_operator = rotate_orbitals(problem, one_body, two_body, matrix_a, matrix_b)
    # overwrite
    global main_op 
    main_op = rotated_operator
    
    try:
        job = solver.estimator.run(solver.ansatz, rotated_operator, parameters)
        estimator_result = job.result()
    except Exception as exc:
        raise KeyError("The primitive job to evaluate the energy failed!") from exc

    values = estimator_result.values
    
    energy = values[0] if len(values) == 1 else values
    
    if callback is not None:
        callback.append(energy)

    # the rest of the energy evaluation code only involves the ansatz parameters

    return energy




# PER QUANDO frozen NON VIENE RICONOSCIUTO: 

def compute_fci_energy_and_integrals (mol, frozen: list):
    
    '''
    Returns:
    - fci energy
    - one body ints on active space
    - two body ints on active space
    '''

    # Run Hartree-Fock
    mf = scf.RHF(mol) # campo medio
    mf.kernel()

    # Freeze the listed orbitals 
    n_orbitals = mf.mo_coeff.shape[1]

    # Get molecular orbital integrals from molecular coefficients and hamiltonian operator
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mf._eri, mf.mo_coeff)  # two electrons integrals

    # Active space: Remove frozen orbitals
    active_orbitals = [i for i in range(n_orbitals) if i not in frozen]
    n_active = len(active_orbitals)

    # Get active space integrals
    h1e_active = h1e[np.ix_(active_orbitals, active_orbitals)]
    eri_active = ao2mo.restore(1, eri, n_orbitals)  # Restore two-electron integrals
    eri_active = eri_active[
        np.ix_(active_orbitals, active_orbitals, active_orbitals, active_orbitals)
    ]
    
    # Numero totale di elettroni
    n_electrons_total = mol.nelec[0] + mol.nelec[1]

    # Numero di elettroni congelati (orbitali moltiplicati per 2 per tenere conto degli spin)
    n_frozen_electrons = len(frozen) * 2

    # Numero di elettroni attivi
    n_active_electrons = n_electrons_total - n_frozen_electrons
    
    # CHECK:
    print('numero elettroni:\n', 'totale: ', n_electrons_total)
    print('freezati: ', n_frozen_electrons, '\n', 'attivi: ', n_active_electrons)
    
    '''
    # Energia di core (orbitali congelati)
    eri_4d = ao2mo.restore(1, eri, n_orbitals)  # Converti in 4D
    core_orbitals = np.ix_(frozen, frozen)
    ecore_one = np.sum(h1e[core_orbitals])  # contributo a 1 elettrone
    ecore_two = 0.5 * np.sum(eri_4d[np.ix_(*[frozen]*4)])  # contributo a 2 elettroni
    ecore = ecore_one + ecore_two + mf.energy_nuc()  # somma con energia nucleare
    '''
    
    # Compute FCI
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel(h1e_active, eri_active, n_active, n_active_electrons) 
    
    return e_fci, h1e_active, eri_active