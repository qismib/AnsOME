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
import warnings
from time import time
from functools import partial 

#from qiskit.aqua import AquaError
#from qiskit.aqua.algorithms import VQE, MinimumEigensolver
#from qiskit.aqua.operators import LegacyBaseOperator

from qiskit.exceptions import QiskitError
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_algorithms import MinimumEigensolver

# from qiskit.opflow import OperatorBase
# deprecated, migration guide URL:
# https://docs.quantum.ibm.com/migration-guides/qiskit-opflow-module

from qiskit.quantum_info import SparsePauliOp 
from qiskit.quantum_info.operators.base_operator import BaseOperator # ...

from qiskit_nature.second_q.algorithms.ground_state_solvers.ground_state_eigensolver import GroundStateEigensolver
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult, VQEResult
from qiskit_nature.second_q.circuit.library.ansatzes.ucc import UCC
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators import BosonicOp
from qiskit_nature.second_q.drivers import BaseDriver # ...
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicBasis # non si evidenzia ma son sicuro che esiste
from qiskit_nature.second_q.problems import ElectronicStructureProblem, ElectronicStructureResult
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.formats.qcschema_translator import get_ao_to_mo_from_qcschema

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo # sostituisco con ElectronicStructureProblem

from qiskit_nature.second_q.mappers import JordanWignerMapper # ...
from qiskit_nature.second_q.mappers import QubitMapper

from oovqe_solver_mio import compute_minimum_eigenvalue_oo
from orbital_rotation_mio import OrbitalRotation

# from qiskit_nature.second_q.mappers import QubitConverter 
# deprecated, just use QubitMapper's .map() method instead, migration guide URL:
# https://qiskit-community.github.io/qiskit-nature/migration/0.6_c_qubit_converter.html


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
# ---- OrbitalOptimizationVQE ---- OrbitalOptimizationVQE ---- OrbitalOptimizationVQE ---- OrbitalOptimizationVQE ----|
#---------------------------------------------------------------------------------------------------------------------|

class OrbitalOptimizationVQE(GroundStateEigensolver):
    """Solver for ooVQE"""

    def __init__(
        self,
        mapper: QubitMapper,
        solver: Union[MinimumEigensolver] = None,
        initial_point: Union[ListOrDict, np.ndarray] = None,
    ) -> None:
        super().__init__(mapper, solver)

        # Store problem to have access during energy eval. function.
        self.driver: PySCFDriver = None
        self.problem: ElectronicStructureProblem = None
        self.problem_rotated: ElectronicStructureProblem = None
        # I am using temporarily the CustomProblem class, that avoids
        # running the driver every time .second_q_ops() is called
        self.orbital_rotation = None  # to set during solve()
        self.num_parameters_oovqe = None  # to set during solve()

        self._mapper = mapper
        self.initial_point = initial_point  #
        self.bounds_oo = None  # in the future: set by user
        self.bounds = None  # ansatz + oo

        self.operator = None

    def set_initial_point(self, initial_pt_scalar: float = 1e-1) -> None:
        """Initializes the initial point for the algorithm if the user does not provide his own.
        Args:
            initial_pt_scalar: value of the initial parameters for wavefunction and orbital rotation
        """
        self.initial_point = np.asarray(
            [initial_pt_scalar for _ in range(self.num_parameters_oovqe)]
        )

    def set_bounds(
        self,
        bounds_ansatz_value: tuple = (-2 * np.pi, 2 * np.pi),
        bounds_oo_value: tuple = (-2 * np.pi, 2 * np.pi),
    ) -> None:
        """Doctstring"""
        bounds_ansatz = [bounds_ansatz_value for _ in range(self.solver.ansatz.num_parameters)]
        self.bounds_oo = [bounds_oo_value for _ in range(self.orbital_rotation.num_parameters)]
        bounds = bounds_ansatz + self.bounds_oo
        self.bounds = np.array(bounds)

    def get_operators(self, problem, aux_operators):
        """Doctstring"""
        
        # aux ops instance  
        aux_second_q_ops: ListOrDict[SparsePauliOp]
        # this return a tuple
        second_q_ops = problem.second_q_ops()
        # its first element is the main op
        main_second_q_op = second_q_ops[0]
        aux_second_q_ops = second_q_ops[1]
            
        main_operator = self._mapper.map(main_second_q_op)
        self.operator = main_operator
        aux_ops = self._mapper.map(aux_second_q_ops)

        # TODO: sistemare aux_ops
        '''
        if aux_operators is not None:
            wrapped_aux_operators: ListOrDict[SparsePauliOp] = ListOrDict(aux_operators)
            for name, aux_op in iter(wrapped_aux_operators):
                if isinstance(aux_op, SparsePauliOp):
                    converted_aux_op = self._qubit_converter.convert_match(aux_op, True)
                else:
                    converted_aux_op = aux_op
                if isinstance(aux_ops, list):
                    aux_ops.append(converted_aux_op)
                elif isinstance(aux_ops, dict):
                    if name in aux_ops.keys():
                        raise QiskitNatureError(
                            f"The key '{name}' is already taken by an internally constructed "
                            "auxiliary operator! Please use a different name for your custom "
                            "operator."
                        )
                    aux_ops[name] = converted_aux_op
        '''
        
        # if the eigensolver does not support auxiliary operators, reset them
        if not self._solver.supports_aux_operators():
            aux_ops = None

        return main_operator, aux_ops

    def rotate_orbitals(self, matrix_a, matrix_b): # questo funziona
        """Doctstring"""
        
        # AO coefficients
        ao_problem      = self.driver.to_problem(basis=ElectronicBasis.AO)
        ao_hamiltonian  = ao_problem.hamiltonian
        ao_coefficients = ao_hamiltonian.electronic_integrals
        
        # Non serve?
        self.problem_rotated = ao_problem
        
        # TODO: capire come funzionano i PolynomialTensor: non supportano assegnazione
        # Rotation - aggiornata per supportare PolynomialTensor
        if ao_coefficients.alpha.is_empty() is not True:
            ao_coefficients.alpha['+-'][:] = np.matmul(ao_coefficients.alpha['+-'], matrix_a)
    
        if ao_coefficients.beta.is_empty() is not True:
            ao_coefficients.beta['+-'][:]  = np.matmul(ao_coefficients.beta['+-'], matrix_b)
        
        elif ao_coefficients.alpha.is_empty() and ao_coefficients.beta.is_empty() is True:
            raise QiskitError('AO coeff. matrices are empty')
        
        # AO one body
        if ao_coefficients.one_body._alpha.is_empty() is not True:
            ao_coefficients.one_body._alpha['+-'][:] = np.matmul(
                ao_coefficients.one_body._alpha['+-'], 
                matrix_a)
            
        if ao_coefficients.one_body._beta.is_empty() is not True:
            ao_coefficients.one_body._beta['+-'][:] = np.matmul(
                ao_coefficients.one_body._beta['+-'], 
                matrix_b) 
            
        elif ao_coefficients.one_body._alpha.is_empty() and ao_coefficients.one_body._beta.is_empty() is True:
            raise QiskitError('AO one body matrices are empty')
        
        # AO to MO transformer
        qcschema = self.driver.to_qcschema()
        basis_transformer = get_ao_to_mo_from_qcschema(qcschema)
        # transformation
        mo_problem = basis_transformer.transform(ao_problem)
        
        rotated_operator = mo_problem.hamiltonian.second_q_op() # lo restituisco e cambio la sintassi della chiamata
        rotated_operator = self._mapper.map(rotated_operator)
        
        # TODO: verificare che anche il dipolo sia incluso
        # dipole = mo_problem.properties.electronic_dipole_moment
        
        return rotated_operator


    def solve(
        self,
        driver: PySCFDriver,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDict[SparsePauliOp]] = None,
    ) -> VQEResult:
        """Compute Ground State properties.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            ValueError: if the grouped property object returned by the driver does not contain a
                main property as requested by the problem being solved (`problem.main_property_name`)
            QiskitNatureError: if the user-provided `aux_operators` contain a name which clashes
                with an internally constructed auxiliary operator. Note: the names used for the
                internal auxiliary operators correspond to the `Property.name` attributes which
                generated the respective operators.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation

        self.driver  = driver
        self.problem = problem

        # IF WE DON'T CALL SECOND_Q_OPS, THE DRIVER DOESN'T RUN AND THE
        # PROPERTIES DON'T GET POPULATED -> CHANGE?
        main_operator, aux_ops = self.get_operators(self.problem, aux_operators)

        self.orbital_rotation = OrbitalRotation(
            num_qubits=self.solver.ansatz.num_qubits, mapper=self._mapper
        )
        self.num_parameters_oovqe = (
            self.solver.ansatz.num_parameters + self.orbital_rotation.num_parameters
        )

        # the initial point of the full ooVQE alg.
        if self.initial_point is None:
            self.set_initial_point()
        else:
            # this will never really happen with the current code
            # but is kept for the future
            if len(self.initial_point) is not self.num_parameters_oovqe:
                raise QiskitNatureError(
                    f"Number of parameters of OOVQE ({self.num_parameters_oovqe,}) "
                    f"does not match the length of the "
                    f"intitial_point ({len(self.initial_point)})"
                )

        if self.bounds is None:
            # set bounds sets both ansatz and oo bounds
            # do we want to change the ansatz bounds here??
            self.set_bounds(self.orbital_rotation.parameter_bound_value)

        # override VQE's compute_minimum_eigenvalue, giving it access to the problem data
        # contained in self.problem
        self.solver.compute_minimum_eigenvalue = partial(
            compute_minimum_eigenvalue_oo, self, self.solver
        )

        raw_mes_result = self.solver.compute_minimum_eigenvalue(main_operator, aux_ops)

        result = problem.interpret(raw_mes_result)

        return result
    
    
    
    


#------------------------------------------------------------------------------------------------------------------