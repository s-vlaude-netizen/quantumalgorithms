### Result 99 — the four-dimensional parallelohedron problem (Cesaroni–Novaga's first open problem): a search over all lattices finds the 24-cell, and a zonotope outside the graphical family beats A₄*

**The question** (Cesaroni & Novaga, arXiv:2609.02384, Open problem 1). Let

    γ₄ = inf over lattices Λ ⊂ ℝ⁴ of  Iso₄(Vor Λ) = (boundary measure) / volume^(3/4).

What is γ₄, and is the regular 24-cell, Vor(D₄) at `2^(11/4)` = 6.72717, the
unique minimiser? In three dimensions the answer is the truncated octahedron
(Hales–Song; Cesaroni–Novaga; Result 98). In four dimensions the proof
technique breaks: not every lattice is of Voronoi's first kind, and there are
52 combinatorial types of parallelohedra.

#### Built — `experiments/exp045_four_dimensional_cells.py`, 7 tests

1. **Exact values** (SymPy), from Gram determinants of each cell's generators
   or facets, each checked against a numerical Voronoi computation:

   | cell | Iso₄ | exact |
   |---|---|---|
   | 24-cell, Vor(D₄) | 6.727171322 | `2^(11/4)` |
   | **cographic K₃,₃ zonotope** | **6.786942232** | **`(32√3 + 4√2)/9`** |
   | permutohedron, Vor(A₄*) | 6.831123656 | `20(16+3√6)/5^(21/8)` |
   | tesseract, Vor(ℤ⁴) | 8 | 8 |

   The cographic K₃,₃ zonotope is the zonotope of the cographic matroid of
   K₃,₃, in its symmetric embedding: the edge basis projected onto the cycle
   space. Every 4×4 minor of that configuration has the same absolute value,
   so it is unimodular and the zonotope tiles ℝ⁴. It is the one zonotopal
   four-dimensional parallelohedron outside the graphical family. A direct
   minimisation over its nine zone weights and SL(4) lands on exactly this
   value from every start; the convex hull agrees to 1e-14.
2. **A random-start search over all four-dimensional lattices.**
   * Each lattice is a lower-triangular basis, LLL-reduced, with its Voronoi
     cell computed by half-space intersection over all vectors with
     coefficients up to ±2.
   * Iso₄ is minimised by Nelder–Mead over the nine shape parameters, from
     LATTICE_STARTS random starts.
3. **The same over all four-dimensional parallelohedra.**
   * By Voronoi's conjecture, proved through dimension five (Garber 2025),
     every parallelohedron is an affine image `A·Vor(Λ)`.
   * For a fixed cell, the surface area after `A ∈ SL(4)` is
     `Σ_k a_k |A^{-T} u_k|`, computed from the facet data alone. The search
     minimises over A inside the lattice search, from AFFINE_STARTS starts.

#### Measured

SEARCH

#### What it says

FINDINGS

#### Grading

GRADE

`experiments/exp045_four_dimensional_cells.py`,
`tests/test_four_dimensional_cells.py`,
`results/exp045_four_dimensional_cells.json`.
