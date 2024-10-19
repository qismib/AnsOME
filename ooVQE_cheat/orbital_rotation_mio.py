# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# QUI L'IDEA È APPLICARE LA TRASFORMAZIONE e^K ALL'HAMILTONIANO

''' 
Refactored by qiskit community. 
https://github.com/qiskit-community/qiskit-nature/pull/629#pullrequestreview-968132514

Updated by zosojack in October 2024 to work with Qiskit Nature 0.7.2
- qiskit.algorithms --> qiskit_algorithms
- qiskit.opflow.OperatorBase --> qiskit.quantum_info.Operator
- qiskit_nature.second_q.mappers.QubitConverter() --> QubitMapper.map()
- transformation --> QubitMapper
- var_form --> ansatz
- VQE.find_minimum() --> VQE.compute_minimum_eigenvalue()
- no longer needed to evaluate cost function before running VQE
   VQE._energy_evaluation() --> Estimator.run()
- QMolecule --> ElectronicStructureProblem
'''
'''
All occurrences of
if isinstance(self._vqe.operator, BaseOperator):  # type: ignore
self._vqe.operator = self._vqe.operator.to_opflow()  # type: ignore
were deleted
'''
'''
Per quanto riguarda _rotate_orbitals_in_molecule()
- https://github.com/qiskit-community/qiskit-nature/blob/af569523110fb88395aa9c17bc0532d9646163f8/docs/tutorials/05_problem_transformers.ipynb
- https://github.com/qiskit-community/qiskit-nature/blob/stable/0.7/docs/tutorials/08_qcschema.ipynb
- in realtà pare che varie manipolazioni elencate di seguito siano sostituite da un paio di righe usando gli oggetti:
  · qcschema_to_problem
  · get_ao_to_mo_from_qcschema
- i coefficienti ora si estraggono dall'oggetto ElectronicEnergy (hamiltonian) che dà un ElectronicIntegrals
  · QMolecule: https://github.com/qiskit-community/qiskit-aqua/blob/stable/0.7/qiskit/chemistry/qmolecule.py
  · ElectronicIntegrals https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.operators.ElectronicIntegrals.html#qiskit_nature.second_q.operators.ElectronicIntegrals
'''

"""
FOR REFERENCE: (credo)
coefficients = problem.hamiltonian.electronic_integrals
- qmolecule.mo_coeff       --> coefficients.alpha
- qmolecule.mo_onee_ints   --> coefficients.one_body.alpha
- qmolecule.mo_coeff_b     --> coefficients.beta
- qmolecule.mo_onee_ints_b --> coefficients.one_body.beta
- qmolecule.mo_eri_ints    --> coefficients.two_body

- qmolecule.x_dip_mo_ints   --> dipole.x_dipole.alpha (= y,z)
- qmolecule.x_dip_mo_ints_b --> dipole.x_dipole.beta (= y,z)
"""

"""
A ground state calculation employing the Orbital-Optimized VQE (OOVQE) algorithm.
"""

from typing import Optional, List, Dict, Union, Tuple, TypeVar
import logging
import copy
import numpy as np
from scipy.linalg import expm

#from qiskit.aqua import AquaError
#from qiskit.aqua.algorithms import VQE, MinimumEigensolver
#from qiskit.aqua.operators import LegacyBaseOperator

from qiskit.exceptions import QiskitError
from qiskit_algorithms import MinimumEigensolver

# from qiskit.opflow import OperatorBase
# deprecated, migration guide URL:
# https://docs.quantum.ibm.com/migration-guides/qiskit-opflow-module

from qiskit.quantum_info import SparsePauliOp 
from qiskit.quantum_info.operators.base_operator import BaseOperator # ...

from qiskit_nature.second_q.algorithms.ground_state_solvers.ground_state_eigensolver import GroundStateEigensolver
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit_nature.second_q.circuit.library.ansatzes.ucc import UCC
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators import BosonicOp
from qiskit_nature.second_q.drivers import BaseDriver # ...
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicBasis # non si evidenzia ma son sicuro che esiste
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.problems import ElectronicStructureResult
from qiskit_nature.second_q.formats.qcschema_translator import get_ao_to_mo_from_qcschema

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo # sostituisco con ElectronicStructureProblem

from qiskit_nature.second_q.mappers import JordanWignerMapper # ...
from qiskit_nature.second_q.mappers import QubitMapper

from qiskit_algorithms.optimizers import COBYLA # ...
from qiskit_algorithms.minimum_eigensolvers import VQE

# Set setting to use SparsePauliOp
import qiskit_nature.settings
qiskit_nature.settings.use_pauli_sum_op = False

from qiskit_nature.second_q.operators import PolynomialTensor

logger = logging.getLogger(__name__)

# Per sostituire ListOrDict
T = TypeVar('T')
ListOrDict = Union[List[T], Dict[str, T]]



#---------------------------------------------------------------------------------------------------------------------|
# ***---- OrbitalRotation ---- OrbitalRotation ---- OrbitalRotation ---- OrbitalRotation ---- OrbitalRotation ----****|
#---------------------------------------------------------------------------------------------------------------------|

class OrbitalRotation:
    r""" Class that regroups methods for creation of matrices that rotate the MOs.
    It allows to create the unitary matrix U = exp(-kappa) that is parameterized with kappa's
    elements. The parameters are the off-diagonal elements of the anti-hermitian matrix kappa.
    """
    # COSTRUTTORE --- COSTRUTTORE --- COSTRUTTORE --- COSTRUTTORE --- COSTRUTTORE --- COSTRUTTORE --- COSTRUTTORE ---
    
    def __init__(self,
                 num_qubits: int,
                 mapper: QubitMapper,
                 problem: Optional[ElectronicStructureProblem] = None,
                 orbital_rotations: list = None,
                 orbital_rotations_beta: list = None,
                 parameters: list = None,
                 parameter_bounds: list = None,
                 parameter_initial_value: float = 0.1,
                 parameter_bound_value: Tuple[float, float] = (-2 * np.pi, 2 * np.pi)) -> None:
        """
        Args:
            num_qubits: number of qubits necessary to simulate a particular system.
            transformation: a fermionic driver to operator transformation strategy.
            qmolecule: instance of the :class:`~qiskit.chemistry.QMolecule` class which has methods
                needed to recompute one-/two-electron/dipole integrals after orbital rotation
                (C = C0 * exp(-kappa)). It is not required but can be used if user wished to
                provide custom integrals for instance.
            orbital_rotations: list of alpha orbitals that are rotated (i.e. [[0,1], ...] the
                0-th orbital is rotated with 1-st, which corresponds to non-zero entry 01 of
                the matrix kappa).
            orbital_rotations_beta: list of beta orbitals that are rotated.
            parameters: orbital rotation parameter list of matrix elements that rotate the MOs,
                each associated to a pair of orbitals that are rotated
                (non-zero elements in matrix kappa), or elements in the orbital_rotation(_beta)
                lists.
            parameter_bounds: parameter bounds
            parameter_initial_value: initial value for all the parameters.
            parameter_bound_value: value for the bounds on all the parameters
        """
        
        # MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI --- MEMBRI ---

        self._num_qubits = num_qubits
        self._mapper = mapper
        self._problem = problem

        self._orbital_rotations = orbital_rotations
        self._orbital_rotations_beta = orbital_rotations_beta
        self._parameter_initial_value = parameter_initial_value
        self._parameter_bound_value = parameter_bound_value
        self._parameters = parameters
        if self._parameters is None:
            self._create_parameter_list_for_orbital_rotations()

        self._num_parameters = len(self._parameters)
        self._parameter_bounds = parameter_bounds
        if self._parameter_bounds is None:
            self._create_parameter_bounds()

        # lascio solo condizione else
        self._dim_kappa_matrix = int(self._num_qubits / 2)
        
        self._check_for_errors()
        self._matrix_a = None
        self._matrix_b = None
        
    # METODI --- METODI --- METODI --- METODI --- METODI --- METODI --- METODI --- METODI --- METODI --- METODI ---

    # controllo iniziale
    def _check_for_errors(self) -> None:
        """ Checks for errors such as incorrect number of parameters and indices of orbitals. """

        # number of parameters check
        if self._orbital_rotations_beta is None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) != len(self._parameters):
                raise QiskitError('Please specify same number of params ({}) as there are '
                                'orbital rotations ({})'.format(len(self._parameters),
                                                                len(self._orbital_rotations)))
        elif self._orbital_rotations_beta is not None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) + len(self._orbital_rotations_beta) != len(
                    self._parameters):
                raise QiskitError('Please specify same number of params ({}) as there are '
                                'orbital rotations ({})'.format(len(self._parameters),
                                                                len(self._orbital_rotations)))
        # indices of rotated orbitals check
        for exc in self._orbital_rotations:
            if exc[0] > (self._dim_kappa_matrix - 1):
                raise QiskitError('You specified entries that go outside '
                                'the orbital rotation matrix dimensions {}, '.format(exc[0]))
            if exc[1] > (self._dim_kappa_matrix - 1):
                raise QiskitError('You specified entries that go outside '
                                'the orbital rotation matrix dimensions {}'.format(exc[1]))
        if self._orbital_rotations_beta is not None:
            for exc in self._orbital_rotations_beta:
                if exc[0] > (self._dim_kappa_matrix - 1):
                    raise QiskitError('You specified entries that go outside '
                                    'the orbital rotation matrix dimensions {}'.format(exc[0]))
                if exc[1] > (self._dim_kappa_matrix - 1):
                    raise QiskitError('You specified entries that go outside '
                                    'the orbital rotation matrix dimensions {}'.format(exc[1]))
                    
    def _create_orbital_rotation_list(self) -> None:
        """ Creates a list of indices of matrix kappa that denote the pairs of orbitals that
        will be rotated. For instance, a list of pairs of orbital such as [[0,1], [0,2]]. """

        # TODO: _two_qubit_reduction deprecated, ora si può implementare con il ParityMapper
        '''
        if self._mapper._two_qubit_reduction:
            half_as = int((self._num_qubits + 2) / 2)
        else:
            half_as = int(self._num_qubits / 2)
        '''
        half_as = int(self._num_qubits / 2) # cioè l'istruzione nell'else
        
        self._orbital_rotations = []

        for i in range(half_as):
            for j in range(half_as):
                if i < j:
                    self._orbital_rotations.append([i, j])

    def _create_parameter_list_for_orbital_rotations(self) -> None:
        """ Initializes the initial values of orbital rotation matrix kappa. """

        # creates the indices of matrix kappa and prevent user from trying to rotate only betas
        if self._orbital_rotations is None:
            self._create_orbital_rotation_list()
        elif self._orbital_rotations is None and self._orbital_rotations_beta is not None:
            raise QiskitError('Only beta orbitals labels (orbital_rotations_beta) have been provided.'
                            'Please also specify the alpha orbitals (orbital_rotations) '
                            'that are rotated as well. Do not specify anything to have by default '
                            'all orbitals rotated.')

        if self._orbital_rotations_beta is not None:
            num_parameters = len(self._orbital_rotations + self._orbital_rotations_beta)
        else:
            num_parameters = len(self._orbital_rotations)
        self._parameters = [self._parameter_initial_value for _ in range(num_parameters)]

    def _create_parameter_bounds(self) -> None:
        """ Create bounds for parameters. """
        self._parameter_bounds = [self._parameter_bound_value for _ in range(self._num_parameters)]

    def orbital_rotation_matrix(self, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Creates 2 matrices K_alpha, K_beta that rotate the orbitals through MO coefficient
        C_alpha = C_RHF * U_alpha where U = e^(K_alpha), similarly for beta orbitals. """

        self._parameters = parameters
        k_matrix_alpha = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))
        k_matrix_beta = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))

        # NOTE: CHECK provvisorio
        print('orbital_rotations: ', len(self._orbital_rotations) , '\n', self._orbital_rotations)
        print('self._parameters: ', len(self._parameters) , '\n', self._parameters)
        
        # allows to selectively rotate pairs of orbitals
        if self._orbital_rotations_beta is None:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[i]
        else:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]

            for j, exc in enumerate(self._orbital_rotations_beta):
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[j + len(self._orbital_rotations)]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[j + len(self._orbital_rotations)]

        # lascio solo istruzione else
        self._matrix_a = expm(k_matrix_alpha)
        self._matrix_b = expm(k_matrix_beta)

        return self._matrix_a, self._matrix_b

    
    def _rotate_orbitals_in_molecule(self, driver: PySCFDriver, problem_rotated) -> None:
        """ Rotates the orbitals by applying a modified anti-hermitian matrix
        (orbital_rotation.matrix_a) onto the MO coefficients matrix and recomputes all the
        quantities dependent on the MO coefficients. Be aware that qmolecule is modified
        when this executes.
        Args:
            qmolecule: instance of QMolecule class
            orbital_rotation: instance of OrbitalRotation class
        """
        
        ''' To add the nuclear component 
        nuclear_dip = dipole.nuclear_dipole_moment
        dipole.x_dipole.alpha += PolynomialTensor({"": nuclear_dip[0]})
        dipole.y_dipole.alpha += PolynomialTensor({"": nuclear_dip[1]})
        dipole.z_dipole.alpha += PolynomialTensor({"": nuclear_dip[2]})
        '''
        
        # per chiarezza
        orbital_rotation = self
        
        # AO coefficients
        ao_problem      = driver.to_problem(basis=ElectronicBasis.AO)
        ao_hamiltonian  = ao_problem.hamiltonian
        ao_coefficients = ao_hamiltonian.electronic_integrals
        
        
        # TODO: capire come funzionano i PolynomialTensor: non supportano assegnazione
        # Rotation - aggiornata per supportare PolynomialTensor
        if ao_coefficients.alpha.is_empty() is not True:
            ao_coefficients.alpha['+-'][:] = np.matmul(ao_coefficients.alpha['+-'], orbital_rotation.matrix_a)
    
        if ao_coefficients.beta.is_empty() is not True:
            ao_coefficients.beta['+-'][:]  = np.matmul(ao_coefficients.beta['+-'],  orbital_rotation.matrix_b)
        
        elif ao_coefficients.alpha.is_empty() and ao_coefficients.beta.is_empty() is True:
            raise QiskitError('AO coeff. matrices are empty')
        
        # AO to MO transformer
        qcschema = driver.to_qcschema()
        basis_transformer = get_ao_to_mo_from_qcschema(qcschema)
        # transformation
        mo_problem = basis_transformer.transform(ao_problem)
        
        return mo_problem # lo restituisco e cambio la sintassi della chiamata
        
        # TODO: verificare che anche il dipolo sia incluso
        # dipole = mo_problem.properties.electronic_dipole_moment
        
        
        
    @property
    def matrix_a(self) -> np.ndarray:
        """Returns matrix A."""
        return self._matrix_a

    @property
    def matrix_b(self) -> np.ndarray:
        """Returns matrix B. """
        return self._matrix_b

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters."""
        return self._num_parameters

    @property
    def parameter_bound_value(self) -> Tuple[float, float]:
        """Returns a value for the bounds on all the parameters."""
        return self._parameter_bound_value

