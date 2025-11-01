# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adaptive optimizer"""

import copy
from typing import Sequence, Callable

try:
    from pennylane import numpy as pnp
    from pennylane.tape import QuantumTape
    from pennylane import transform
except ImportError as e:
    raise ImportError(f"Install pennylane for using the pennylane ADAPT implementation. {e}")
import excitation_solve


@transform
def append_gate(tape: QuantumTape, params, gates) -> (Sequence[QuantumTape], Callable):
    """Append parameterized gates to an existing tape.

    Args:
        tape (QuantumTape or QNode or Callable): quantum circuit to transform by adding gates
        params (array[float]): parameters of the gates to be added
        gates (list[Operator]): list of the gates to be added

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    """
    new_operations = []

    for i, g in enumerate(gates):
        g = copy.copy(g)
        new_params = (params[i], *g.data[1:])
        g.data = new_params
        new_operations.append(g)

    new_tape = type(tape)(tape.operations + new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing


class ExcitationAdaptiveOptimizer:
    r"""Optimizer for building fully trained quantum circuits by adding gates adaptively using ExcitationSolve."""

    @staticmethod
    def _circuit(params, gates, initial_circuit):
        """Append parameterized gates to an existing circuit.

        Args:
            params (array[float]): parameters of the gates to be added
            gates (list[Operator]): list of the gates to be added
            initial_circuit (function): user-defined circuit that returns an expectation value

        Returns:
            function: user-defined circuit with appended gates
        """
        final_circuit = append_gate(initial_circuit, params, gates)

        return final_circuit()

    def step(self, circuit, operator_pool, params_zero=True):
        r"""Update the circuit with one step of the optimizer.

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            params_zero (bool): flag to initiate circuit parameters at zero

        Returns:
            .QNode: the optimized circuit
        """
        return self.step_and_cost(circuit, operator_pool, params_zero=params_zero)[0]

    def step_and_cost(self, circuit, operator_pool, drain_pool=False, params_zero=False):
        r"""Update the circuit with one step of the optimizer, return the corresponding
        objective function value prior to the step, and return the maximum gradient

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            drain_pool (bool): flag to remove selected gates from the operator pool
            params_zero (bool): flag to initiate circuit parameters at zero

        Returns:
            tuple[.QNode, float, float]: the optimized circuit, the objective function output prior
            to the step, and the largest gradient
        """
        cost = circuit()
        qnode = copy.copy(circuit)

        if drain_pool:
            operator_pool = [
                gate
                for gate in operator_pool
                if all(gate.name != operation.name or gate.wires != operation.wires for operation in circuit.tape.operations)
            ]

        params = pnp.array([gate.parameters[0] for gate in operator_pool], requires_grad=True)
        qnode.func = self._circuit

        shifts = pnp.array([0, pnp.pi / 2, -pnp.pi / 2, pnp.pi, -pnp.pi])
        energies = []
        opt_params = []
        # Select best operator to append based on ExcitationSolve
        for param_to_vary, op in enumerate(operator_pool):
            e_shifted = [
                qnode(
                    [shift],
                    gates=[op],
                    initial_circuit=circuit.func,
                )
                for shift in shifts
            ]

            opt_param, e_excsolve = excitation_solve.excitation_solve_step(shifts, e_shifted)
            opt_params.append(opt_param)
            energies.append(e_excsolve)
        energies = pnp.array(energies)
        selected_index = pnp.argmax(cost - energies)
        selected_gates = [operator_pool[selected_index]]

        # set parameter of appended op to its optimized value
        if params_zero:
            params = pnp.zeros(len(selected_gates))
        else:
            params = pnp.array(
                [opt_params[selected_index]],
                requires_grad=True,
            )

        qnode.func = append_gate(circuit.func, params, selected_gates)

        return qnode, cost, max(abs(cost - energies))
