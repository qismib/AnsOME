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

from typing import Optional, List, Dict, Union, Tuple, TypeVar, Callable
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

from qiskit.exceptions            import QiskitError
from qiskit_nature.exceptions     import QiskitNatureError
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms import MinimumEigensolver

# from qiskit.opflow import OperatorBase
# deprecated, migration guide URL:
# https://docs.quantum.ibm.com/migration-guides/qiskit-opflow-module

from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator # ...
from qiskit.circuit import QuantumCircuit

from qiskit_nature.second_q.algorithms.ground_state_solvers.ground_state_eigensolver import GroundStateEigensolver
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult, VQEResult
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

# from qiskit_nature.second_q.mappers import QubitConverter 
# deprecated, just use QubitMapper's .map() method instead, migration guide URL:
# https://qiskit-community.github.io/qiskit-nature/migration/0.6_c_qubit_converter.html


from qiskit_algorithms.optimizers import COBYLA #

from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.observables_evaluator import estimate_observables
from qiskit_algorithms.gradients import LinCombEstimatorGradient

# Set setting to use SparsePauliOp
import qiskit_nature.settings
qiskit_nature.settings.use_pauli_sum_op = False

from qiskit_nature.second_q.operators import PolynomialTensor

logger = logging.getLogger(__name__)

# Per sostituire ListOrDict
T = TypeVar('T')
ListOrDict = Union[List[T], Dict[str, T]]

#---------------------------------------------------------------------------------------------------------------------|
# *****---------- SOLVER ----------- SOLVER ----------- SOLVER ----------- SOLVER ----------- SOLVER -----------******|
#---------------------------------------------------------------------------------------------------------------------|
        
# Possible way to account for orbital params
# NOTE: sto barando: finché è un simulatore va bene, l'exp value andrebbe calcolato con quantum hardware
def numerical_orbital_gradient(ground_state_eigensolver, solver, params, delta=1e-5):
    
    ansatz_params   = params[:solver.ansatz.num_parameters]
    rotation_params = params[solver.ansatz.num_parameters:]
    # initialize grad
    grad = np.zeros(len(rotation_params)) 

    for i in range(len(rotation_params)):
        # perturb i-th parameter
        params_up = np.copy(rotation_params)
        params_down = np.copy(rotation_params)
        
        params_up[i] += delta
        params_down[i] -= delta
        
        # perturbed rotation matrices
        matrix_a_up, matrix_b_up     = ground_state_eigensolver.orbital_rotation.orbital_rotation_matrix(params_up)
        matrix_a_down, matrix_b_down = ground_state_eigensolver.orbital_rotation.orbital_rotation_matrix(params_down)

        # perturbed operators
        rotated_operator_up   = ground_state_eigensolver.rotate_orbitals(matrix_a_up, matrix_b_up)
        rotated_operator_down = ground_state_eigensolver.rotate_orbitals(matrix_a_down, matrix_b_down)

        # expectation values
        job_up   = solver.estimator.run(solver.ansatz, rotated_operator_up, ansatz_params)
        job_down = solver.estimator.run(solver.ansatz, rotated_operator_down, ansatz_params)

        estimator_result_up   = job_up.result().values[0]
        estimator_result_down = job_down.result().values[0]
        
        # gradient approximation by finite difference 
        grad[i] = (estimator_result_up - estimator_result_down) / (2 * delta)
        
    return grad


# EXPECTATION VALUE 
def energy_evaluation_oo(
    ground_state_eigensolver, solver, parameters: np.ndarray
) -> Union[float, List[float]]:
    """Doctstring"""
    print("parameters: ", parameters)
    num_parameters_ansatz = solver.ansatz.num_parameters
    if num_parameters_ansatz == 0:
        raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

    ansatz_params = solver.ansatz.parameters

    ansatz_parameter_values = parameters[:num_parameters_ansatz]
    rotation_parameter_values = parameters[num_parameters_ansatz:] 

    print('Parameters of wavefunction are: \n%s', repr(ansatz_parameter_values))
    print('Parameters of orbital rotation are: \n%s', repr(rotation_parameter_values))

    # CALCULATE COEFFICIENTS OF ROTATION MATRIX HERE:
    matrix_a, matrix_b = ground_state_eigensolver.orbital_rotation.orbital_rotation_matrix(
        rotation_parameter_values
    )

    print("Nature matrix a: ", matrix_a)
    print("Nature matrix b: ", matrix_b)
    # ROTATE AND RECOMPUTE OPERATOR HERE:
    rotated_operator = ground_state_eigensolver.rotate_orbitals(matrix_a, matrix_b)

    print("Rotated operator: ", rotated_operator[:5])
    # use rotated operator for constructing expect_op
    # TODO: Estimator variance?
    # - solver.construct_expectation
    # - solver._circuit_sampler.convert
    # - expectation.compute_variance
    
    start_time = time()
    
    try:
            job = solver.estimator.run(solver.ansatz, rotated_operator, ansatz_parameter_values)
            estimator_result = job.result()
    except Exception as exc:
        raise KeyError("The primitive job to evaluate the energy failed!") from exc

    values = estimator_result.values
    
    energy = values[0] if len(values) == 1 else values
    
    end_time = time()
    # the rest of the energy evaluation code only involves the ansatz parameters
    
    print(
        "Energy evaluation returned %s - %.5f (ms), eval count: %s",
        energy,
        (end_time - start_time) * 1000,
        #solver._eval_count,
    )
    logger.info(
        "Energy evaluation returned %s - %.5f (ms), eval count: %s",
        energy,
        (end_time - start_time) * 1000,
        #solver._eval_count,
    )

    return energy

def get_evaluate_gradient_oo(
        ground_state_eigensolver, # HACK: per provare il gradiente numerico orbitale
        solver,
        ansatz: QuantumCircuit,
        operator: BaseOperator,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A function handle to evaluate the gradient at given parameters for the ansatz.

        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """
        # only the ansatz (parametric circuit) gradient 
        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            
            # only ansatz parameters 
            ansatz_params = parameters[:ansatz.num_parameters]
            # broadcasting not required for the estimator gradients
            try:
                job = solver.gradient.run(
                    [ansatz], [operator], [ansatz_params]  # type: ignore[list-item]
                )
                gradients = job.result().gradients
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            # FIXME: optimizer wants a gradient with len(parameters) entries, but gradient returned by VQE method has 
            # ansatz.num_parameters. This prevents minimizer from failing, though fixes orbital parameters
            
            grad = gradients[0]
            
            # HACK: filling the rest with zeros
            full_gradient = np.zeros(len(parameters))
            full_gradient[:ansatz.num_parameters] = grad
            
            # HACK: expanding grad with numerical gradient for orbital params
            full_gradient[solver.ansatz.num_parameters:] = numerical_orbital_gradient(
                ground_state_eigensolver, 
                solver,
                parameters)

            
            return full_gradient

        return evaluate_gradient

# global variable
count = 0 

# QUESTO È IL FOCUS: COME FUNZIONA? COSA FA PER CALCOLARE I K?
def compute_minimum_eigenvalue_oo(
    ground_state_eigensolver,
    solver,
    operator: BaseOperator,
    aux_operators: Optional[ListOrDict[BaseOperator]] = None,
) -> MinimumEigensolverResult:
    """Doctstring"""

    # this sets the size of the ansatz, so it must be called before the initial point
    # validation
    solver._check_operator_ansatz(operator)

    # # set an expectation for this algorithm run (will be reset to None at the end)
    # initial_point_ansatz = _validate_initial_point(solver.initial_point, solver.ansatz)
    # bounds_ansatz = _validate_bounds(solver.ansatz)
    #
    # # HERE: the real initial point and bounds include the ansatz and the oo parameters:
    # bounds_oo_val: tuple = (-2 * np.pi, 2 * np.pi)
    # initial_pt_scalar: float = 1e-1
    #
    # initial_point_oo = np.asarray(
    #     [initial_pt_scalar for _ in range(ground_state_eigensolver.orbital_rotation.num_parameters)]
    # )
    # bounds_oo = np.asarray(
    #     [bounds_oo_val for _ in range(ground_state_eigensolver.orbital_rotation.num_parameters)]
    # )

    initial_point = ground_state_eigensolver.initial_point
    bounds = ground_state_eigensolver.bounds

    # np.concatenate((bounds_ansatz, bounds_oo))

    # HERE: for the moment, not taking care of aux_operators
    # Does the orbital rotation affect them???
    # We need to handle the array entries being zero or Optional i.e. having value None
    if aux_operators:
        zero_op = SparsePauliOp.from_list([("I" * solver.ansatz.num_qubits, 0)])

        # Convert the None and zero values when aux_operators is a list.
        # Drop None and convert zero values when aux_operators is a dict.
        if isinstance(aux_operators, list):
            key_op_iterator = enumerate(aux_operators)
            converted = [zero_op] * len(aux_operators)
        else:
            key_op_iterator = aux_operators.items()
            converted = {}
        for key, op in key_op_iterator:
            if op is not None:
                converted[key] = zero_op if op == 0 else op

        aux_operators = converted

    else:
        aux_operators = None

    # NOTE:
    # 1. it is required for the solver to have a gradient instance
    # 2. optimizer.minimize() passes the entire param vector to the gradient, but only ansatz parameters are needed
    #    POSSIBLE SOLUTIONS
    #    2.1) custom function _get_evaluate_gradient_oo 
    #    2.2) optimizer.wrap_function(function,params)
    if solver.gradient: # HACK: ground_state_eigensolver argomento provvisorio
        # 2.1) works
        gradient = get_evaluate_gradient_oo(ground_state_eigensolver, solver, solver.ansatz, operator)
        # 2.2) doesn't work
        # N = solver.ansatz.num_parameters
        # gradient = solver.optimizer.wrap_function(solver._get_evaluate_gradient, initial_point[:N])
    else:
        raise QiskitNatureError('The solver must contain a gradient instance')
    
    # FIXME: non esiste più, forse c'è un alternativa?
    # solver._eval_count = 0

    # HERE: custom energy eval. function to pass to optimizer
    energy_evaluation = partial(energy_evaluation_oo, ground_state_eigensolver, solver)

    start_time = time()

    # minimization
    opt_result = solver.optimizer.minimize(
        fun=energy_evaluation, x0=initial_point, jac=gradient, bounds=bounds
    )
    
    eval_time = time() - start_time

    result = VQEResult()
    result.optimal_point = opt_result.x
    result.optimal_parameters = dict(zip(solver.ansatz.parameters, opt_result.x))
    result.optimal_value = opt_result.fun
    result.cost_function_evals = opt_result.nfev
    result.optimizer_time = eval_time
    result.eigenvalue = opt_result.fun + 0j
   
    logger.info(
        "Optimization complete in %s seconds.\nFound opt_params %s in %s evals",
        eval_time,
        result.optimal_point,
        #solver._eval_count,
    )
    
    # estimate_observables only needs ansatz parameters
    ansatz_params = opt_result.x[:solver.ansatz.num_parameters]
    
    if aux_operators is not None:
            aux_operators_evaluated = estimate_observables(
                solver.estimator,
                solver.ansatz,
                aux_operators,
                ansatz_params,  # type: ignore[arg-type]
            )
    else:
        aux_operators_evaluated = None

    result.aux_operator_eigenvalues = aux_operators_evaluated
    
    print("result: ", result)
    
    global count 
    count += 1    
    
    print("[ ", count, " ] -------------------------------------------------------------------------------------------------- ")

    return result
