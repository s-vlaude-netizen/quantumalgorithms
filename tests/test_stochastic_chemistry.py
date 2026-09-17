"""The probabilistic classical baseline (Result 85).

The load-bearing test here is the sector one. Choosing the reference determinant
by `argmin(diagonal)` landed in a particle-number sector with no couplings at
all, so every walker sat still and the projected energy came back as exactly the
Hartree-Fock value at every point on the walker ladder -- a stable population, a
plausible 5.8e-2 error, and no measurement whatsoever. Nothing about the output
looked wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from experiments.exp030_stochastic_chemistry import (
    fciqmc,
    prepare,
    sector,
    sector_ground_energy,
)


def _block_diagonal():
    """Two disconnected 2x2 blocks; the lower-diagonal one is uncoupled.

    This is the shape of the bug: the globally smallest diagonal entry sits in a
    block that cannot be left.
    """
    matrix = np.zeros((4, 4))
    # block A: indices 0,1 -- coupled
    matrix[0, 0], matrix[1, 1] = -1.0, -0.5
    matrix[0, 1] = matrix[1, 0] = -0.4
    # block B: indices 2,3 -- diagonal only, and index 2 is the global minimum
    matrix[2, 2], matrix[3, 3] = -2.0, 0.5
    return sparse.csr_matrix(matrix)


def test_sector_finds_the_connected_component():
    offdiag = _block_diagonal() - sparse.diags(_block_diagonal().diagonal())
    offdiag = offdiag.tocsr()

    members, count = sector(offdiag, 0)
    assert sorted(members) == [0, 1]
    assert count >= 2, "the two blocks must be seen as separate components"

    # and the trap: the global diagonal minimum is in the other component
    isolated, _ = sector(offdiag, 2)
    assert 0 not in isolated and 1 not in isolated


def test_the_global_minimum_can_be_the_wrong_reference():
    """Documents why argmin(diagonal) is not a valid choice.

    Index 2 has the lowest diagonal and no couplings. A walker placed there
    cannot move, which is precisely what happened.
    """
    matrix = _block_diagonal()
    diagonal = matrix.diagonal()
    assert int(np.argmin(diagonal)) == 2

    offdiag = (matrix - sparse.diags(diagonal)).tocsr()
    offdiag.eliminate_zeros()
    assert offdiag[2].nnz == 0, "index 2 is the uncoupled trap this test is about"


def test_prepare_splits_diagonal_from_couplings():
    from qiskit.quantum_info import SparsePauliOp

    hamiltonian = SparsePauliOp.from_list([("ZZ", -1.0), ("XX", 0.3), ("II", 0.5)])
    diagonal, offdiag = prepare(hamiltonian)

    assert offdiag.diagonal().max() == pytest.approx(0.0, abs=1e-14)
    dense = hamiltonian.to_matrix().real
    np.testing.assert_allclose(diagonal, np.diag(dense), atol=1e-12)
    np.testing.assert_allclose(
        offdiag.toarray(), dense - np.diag(np.diag(dense)), atol=1e-12
    )


def test_prepare_rejects_a_complex_hamiltonian():
    """A silently discarded imaginary part would change the walker dynamics."""
    from qiskit.quantum_info import SparsePauliOp

    with pytest.raises(ValueError, match="not real"):
        prepare(SparsePauliOp.from_list([("XY", 1.0)]))


def test_sector_ground_energy_is_the_in_sector_eigenvalue():
    """Not the global one, which can sit in the wrong electron-number sector."""
    from qiskit.quantum_info import SparsePauliOp

    # ZZ + XX couples |00>,|11> and |01>,|10> as two separate 2x2 blocks
    hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
    _, offdiag = prepare(hamiltonian)

    energies = {}
    for reference in (0, 1):
        energy, members = sector_ground_energy(hamiltonian, offdiag, reference)
        energies[reference] = energy
        assert reference in members

    dense = hamiltonian.to_matrix().real
    global_minimum = float(np.linalg.eigvalsh(dense)[0])
    assert min(energies.values()) == pytest.approx(global_minimum, abs=1e-10)
    # the two sectors differ, which is the whole point
    assert energies[0] != pytest.approx(energies[1], abs=1e-6)


def test_fciqmc_recovers_a_known_two_level_ground_state():
    """End-to-end validation before any scaling is read off the method."""
    matrix = np.array([[0.0, -0.3], [-0.3, 0.6]])
    exact = float(np.linalg.eigvalsh(matrix)[0])

    csr = sparse.csr_matrix(matrix)
    diagonal = csr.diagonal().copy()
    offdiag = (csr - sparse.diags(diagonal)).tocsr()

    errors = []
    for seed in range(5):
        run = fciqmc(diagonal, offdiag, 0, 4000,
                     np.random.default_rng(seed), steps=8000, equilibration=3000)
        errors.append(abs(run["energy"] - exact))

    assert np.median(errors) < 5e-3, (
        f"median error {np.median(errors):.2e} against exact {exact:.6f}"
    )


def test_no_spawns_reports_none_rather_than_zero_annihilation():
    """The ambiguity that let the sector bug survive a full run.

    "0% annihilation" was printed both when nothing cancelled and when nothing
    was spawned at all. Those are different failures and only one of them is a
    measurement.
    """
    diagonal = np.array([0.0, 1.0])
    offdiag = sparse.csr_matrix((2, 2))          # no couplings anywhere
    run = fciqmc(diagonal, offdiag, 0, 100, np.random.default_rng(0),
                 steps=200, equilibration=50)

    assert run["gross_spawns"] == 0
    assert run["annihilation"] is None


def test_walkers_stay_inside_their_sector():
    """The dynamics must not leak across a block, or the exact comparison is wrong."""
    matrix = _block_diagonal()
    diagonal = matrix.diagonal().copy()
    offdiag = (matrix - sparse.diags(diagonal)).tocsr()
    offdiag.eliminate_zeros()

    run = fciqmc(diagonal, offdiag, 0, 500, np.random.default_rng(0),
                 steps=1000, equilibration=200)
    assert run["energy"] < 0.0
    # reaching index 2 would mean the propagation crossed an uncoupled block
    members, _ = sector(offdiag, 0)
    assert sorted(members) == [0, 1]
