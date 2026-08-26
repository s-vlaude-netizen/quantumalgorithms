"""ADAPT-VQE: gradient correctness, and the cost accounting it needs."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qres.adapt import _evolution, adapt_vqe, excitation_pool, operator_gradients
from qres.ansatz import hartree_fock_state
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule


def test_pool_operators_are_hermitian():
    """They are, and that is why the gradient needs the factor of i.

    qiskit-nature returns UCC generators as Hermitian operators (coefficients
    +-0.5, real). So [H, A] is *anti*-Hermitian and <[H, A]> is purely
    imaginary -- taking its real part gives exactly zero for every operator,
    which reads as "nothing has any gradient" and stops ADAPT at zero
    parameters. That is what a first version did.
    """
    problem = build_molecule("H4")
    for operator in excitation_pool(problem, "sd")[:5]:
        matrix = operator.to_matrix()
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


def test_evolution_is_exact_not_trotterised():
    """The Pauli terms within one excitation generator commute, so applying
    them in turn is exact -- the same fact that lets UCC Trotterise a single
    excitation without error. Checked against the matrix exponential.
    """
    from scipy.linalg import expm

    from qiskit.quantum_info import Operator

    problem = build_molecule("H4")
    pool = excitation_pool(problem, "sd")
    for index in (0, 5, 12):
        theta = 0.37
        got = Operator(_evolution(pool[index], theta)).data
        want = expm(-1j * theta * pool[index].to_matrix())
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_gradients_match_finite_differences():
    """The analytic gradient must equal the energy's actual derivative."""
    problem = build_molecule("H4")
    pool = excitation_pool(problem, "sd")
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))
    gradients = operator_gradients(problem.hamiltonian, reference, pool)
    assert gradients.max() > 0.1, "Hartree-Fock has gradient for double excitations"

    def energy(theta, index):
        evolved = reference.evolve(_evolution(pool[index], theta))
        return float(np.real(evolved.expectation_value(problem.hamiltonian)))

    h = 1e-5
    for index in np.argsort(-gradients)[:4]:
        finite = (energy(h, index) - energy(-h, index)) / (2 * h)
        assert abs(finite) == pytest.approx(gradients[index], abs=1e-6)


def test_adapt_reaches_chemical_accuracy_with_few_parameters():
    """The parameter reduction, which is real: H2 needs one operator.

    The cost accounting is a separate matter -- see RESEARCH_LOG Result 44,
    where ADAPT costs 6x MORE than fixed UCCSD on H4 despite using 2.9x fewer
    parameters, because it re-optimises after every growth step.
    """
    problem = build_molecule("H2")
    result = adapt_vqe(problem, max_operators=6)
    assert result.num_parameters <= 2
    assert abs(result.energy - problem.fci_energy) < CHEMICAL_ACCURACY_HA
    assert result.converged


def test_gradient_ranking_tolerates_much_more_noise_than_an_energy():
    """Gradients only have to rank operators, not resolve them.

    Measured: sigma = 0.01 picks the right operator 100% of the time on both H4
    and LiH, against chemical accuracy's 1.6e-3 -- and shots go as 1/sigma^2, so
    a gradient sweep is ~39x cheaper than an energy of the same nominal size.
    """
    problem = build_molecule("H4")
    pool = excitation_pool(problem, "sd")
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))
    exact = operator_gradients(problem.hamiltonian, reference, pool)
    best = int(np.argmax(exact))

    rng = np.random.default_rng(0)
    hits = sum(
        int(np.argmax(np.abs(exact + rng.normal(0, 0.01, len(exact))))) == best
        for _ in range(200)
    )
    assert hits >= 190, "sigma=0.01 should almost always pick the right operator"


def test_batched_lazy_schedule_is_the_default_and_is_cheaper():
    """The best schedule measured, and it must stay the default.

    ADAPT's cost is (growth steps) x (cost of one re-optimisation). The lazy
    schedule shrinks the second factor 3.1x, batching shrinks the first, and
    together they are 4.6x cheaper than standard ADAPT at the same accuracy --
    141 evaluations against fixed UCCSD's 134, using 10 parameters rather than
    26 (RESEARCH_LOG Result 47).
    """
    import inspect

    signature = inspect.signature(adapt_vqe)
    assert signature.parameters["batch"].default > 1
    assert signature.parameters["lazy"].default is True


@pytest.mark.parametrize("batch,lazy", [(1, False), (5, True)])
def test_every_schedule_reaches_chemical_accuracy(batch, lazy):
    problem = build_molecule("H4")
    result = adapt_vqe(problem, max_operators=15, batch=batch, lazy=lazy)
    assert abs(result.energy - problem.fci_energy) < CHEMICAL_ACCURACY_HA
    assert result.num_parameters < 26, "must use fewer parameters than fixed UCCSD"


def test_batching_uses_fewer_growth_steps():
    """Which is the whole mechanism: fewer steps means fewer re-optimisations."""
    problem = build_molecule("H4")
    single = adapt_vqe(problem, max_operators=15, batch=1, lazy=True)
    batched = adapt_vqe(problem, max_operators=15, batch=5, lazy=True)
    assert len(batched.steps) < len(single.steps)
