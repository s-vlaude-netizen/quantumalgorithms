"""The repetition question (Result 75), and the caveat that decides its size.

Result 71 left the error-corrected estimate with one unbounded quantity: how many
times the circuit must run. The answer turns on a *scaling* difference between
two algorithm families, so the tests pin the scaling rather than any single
number -- a constant that happens to be right today would not protect the
conclusion.

The last test is the important one. The absolute drug-sized figure this
experiment produces is dominated by a deliberately naive block encoding, not by
physics, and it reproduces roughly where the published field stood in 2017.
Quoting it as *the* cost of fault-tolerant chemistry would be exactly the
overclaim Result 72 was written about, with the sign reversed.
"""

from __future__ import annotations

import pytest

from experiments.exp019_repetitions import (
    CHEMICAL_ACCURACY,
    QPE_REPETITIONS,
    qpe_resources,
    vqe_resources,
)

# H4, measured in this repository
LAMBDA = 8.47
TERMS = 165
PARAMETERS = 26
T_PER_CIRCUIT = 7_997


def test_qpe_repetitions_do_not_depend_on_the_target_accuracy():
    """The load-bearing claim: precision goes into depth, not into repeats.

    If this ever stopped holding, the whole answer to Result 71's open question
    would change -- it is the entire reason a fault-tolerant machine helps.
    """
    counts = {
        qpe_resources(LAMBDA, TERMS, epsilon)["repetitions"]
        for epsilon in (1.6e-2, 1.6e-3, 1.6e-6, 1.6e-9)
    }
    assert counts == {QPE_REPETITIONS}


def test_vqe_repetitions_scale_as_one_over_epsilon_squared():
    """And the contrast: VQE puts precision into repetitions, quadratically."""
    loose = vqe_resources(LAMBDA, PARAMETERS, T_PER_CIRCUIT, 1.6e-3)["repetitions"]
    tight = vqe_resources(LAMBDA, PARAMETERS, T_PER_CIRCUIT, 1.6e-4)["repetitions"]
    assert tight / loose == pytest.approx(100.0, rel=1e-9)


def test_qpe_depth_scales_as_one_over_epsilon_not_its_square():
    """Linear, which is what makes the trade favourable."""
    loose = qpe_resources(LAMBDA, TERMS, 1.6e-3)["walks"]
    tight = qpe_resources(LAMBDA, TERMS, 1.6e-4)["walks"]
    assert tight / loose == pytest.approx(10.0, rel=1e-9)


def test_the_advantage_grows_with_precision_rather_than_being_a_constant():
    """A constant factor would not answer the question; a scaling one does."""
    ratios = []
    for epsilon in (1.6e-2, 1.6e-3, 1.6e-4):
        qpe = qpe_resources(LAMBDA, TERMS, epsilon)
        vqe = vqe_resources(LAMBDA, PARAMETERS, T_PER_CIRCUIT, epsilon)
        ratios.append(vqe["t_gates"] / qpe["t_gates"])

    assert ratios[1] / ratios[0] == pytest.approx(10.0, rel=1e-6)
    assert ratios[2] / ratios[1] == pytest.approx(10.0, rel=1e-6)
    assert ratios[0] > 1, "QPE should already win at loose accuracy"


def test_the_advantage_holds_at_chemical_accuracy_on_a_real_molecule():
    """The headline, on H4's measured lambda and term count."""
    qpe = qpe_resources(LAMBDA, TERMS, CHEMICAL_ACCURACY)
    vqe = vqe_resources(LAMBDA, PARAMETERS, T_PER_CIRCUIT, CHEMICAL_ACCURACY)

    assert vqe["t_gates"] / qpe["t_gates"] > 1e4
    assert vqe["repetitions"] > 1e8, "the measured shot law should be brutal here"
    assert qpe["repetitions"] < 10


def test_the_naive_encoding_is_far_behind_the_published_state_of_the_art():
    """The caveat that stops the drug-sized number being quoted as the cost.

    Lee et al. 2021 report 5.3e9 Toffolis for FeMoco with tensor
    hypercontraction. The naive LCU here is orders of magnitude worse, and that
    gap is the Hamiltonian representation, not the hardware. If a future version
    implements a better block encoding this test should be updated rather than
    deleted -- the point is that the comparison is made at all.
    """
    lambda_drug = LAMBDA * (50 / 4) ** 2.86
    terms_drug = TERMS * (50 / 4) ** 4
    naive = qpe_resources(lambda_drug, terms_drug, CHEMICAL_ACCURACY)["t_gates"]

    state_of_the_art = 5.3e9 * 4  # Toffolis -> T gates
    assert naive / state_of_the_art > 1e3, (
        "the naive encoding should be far behind; if it is not, the anchor is wrong"
    )


def test_resources_grow_with_lambda_and_terms():
    """Sanity: both models must be monotone in the inputs they claim to use."""
    base = qpe_resources(LAMBDA, TERMS, CHEMICAL_ACCURACY)["t_gates"]
    assert qpe_resources(2 * LAMBDA, TERMS, CHEMICAL_ACCURACY)["t_gates"] > base
    assert qpe_resources(LAMBDA, 2 * TERMS, CHEMICAL_ACCURACY)["t_gates"] > base

    vqe_base = vqe_resources(LAMBDA, PARAMETERS, T_PER_CIRCUIT, CHEMICAL_ACCURACY)
    doubled = vqe_resources(2 * LAMBDA, PARAMETERS, T_PER_CIRCUIT, CHEMICAL_ACCURACY)
    assert doubled["shots"] / vqe_base["shots"] == pytest.approx(4.0)
