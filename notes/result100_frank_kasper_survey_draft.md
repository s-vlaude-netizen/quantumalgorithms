### Result 99 — the Frank–Kasper rivals of Weaire–Phelan, relaxed with the cell free: none comes close, and the order follows the mean face count

**The question.** Weaire–Phelan is the soap-film version of one crystal
structure, A15, a member of the *Frank–Kasper* family. These are the
tetrahedrally close-packed structures of metallurgy, whose Voronoi cells have
12, 14, 15 or 16 faces, all pentagons and hexagons, with four cells at every
vertex. They are the natural place to look for something better.

What the literature has:
* Phelan: σ and H beat Kelvin but lose to Weaire–Phelan.
* Reddy et al., PNAS 2018: P also beats Kelvin; Z, μ, δ, pσ, M, C14 and C15 do
  not. They fixed the periodic cell at the crystal's lattice ratios.
* Cox et al., J. Phys. Condens. Matter 2017: quasiperiodic Frank–Kasper
  approximants come close to Weaire–Phelan from above.
* Oudet (Exp. Math. 2011): a Γ-convergence search with no initial guess, in a
  periodic *cube* with 8–20 cells, finds Weaire–Phelan best.

#### Built — `experiments/exp043_kelvin_search.py`, 22 tests

* **Structures from crystal data.** Ten structures:
  * A15, σ, Z, μ, P, δ;
  * the Laves polytypes C14, C15 and C36;
  * α-Mn (χ), which is not Frank–Kasper, as a control.

  Each is generated from space-group generators and its asymmetric unit, with
  coordinates from the AFLOW prototype library and the Crystallography Open
  Database. The group closures and Wyckoff multiplicities are checked; centred
  cells are reduced to primitive ones.
* **Census before relaxing.** Every Frank–Kasper foam has only pentagons and
  hexagons, the expected 12/14/15/16 face counts, and four cells per vertex.
  * **One database error found.** COD 2310313 (the P phase) lists its cell
    edges in an order that does not fit its own coordinates: the foam then has
    9- to 20-face cells. The paper's order (9.070, 16.983, 4.752 Å) gives the
    Frank–Kasper foam.
  * **The R phase was dropped.** Its refined, disordered coordinates give 21
    quadrilateral faces, so it is not the Frank–Kasper foam.
* **Relaxation.** The exp042 relaxer:
  * equal volumes, with the periodic box free;
  * levels 1–3, Richardson-extrapolated;
  * the cost is normalised by the cells' actual mean volume, which removes the
    first-order bias of residual volume errors.

  Precision, against the published Kelvin and Weaire–Phelan values: 2e-5 or
  better. The scatter between two orderings of the same cells is 4e-6.

#### Measured

TABLE

FINDINGS

#### Grading

GRADE

`experiments/exp043_kelvin_search.py`, `tests/test_kelvin_search.py`,
`results/exp043_kelvin_search.json`, `results/figures/exp043_frank_kasper_costs.png`.
