"""Experiment 043 -- the Kelvin problem beyond Weaire-Phelan: Frank-Kasper foams.

Weaire-Phelan (exp042) is the Voronoi foam of the A15 crystal structure, one of
the *Frank-Kasper* (tetrahedrally close-packed) structures of metallurgy. Its
cells have 12, 14, 15 or 16 faces, with only pentagonal and hexagonal faces;
every vertex has four cells, as Plateau's laws want. The other Frank-Kasper
structures are therefore the natural rivals, and two of them (sigma and H,
Phelan) and a third (P; Reddy et al., PNAS 115, 10233 (2018)) are known to beat
Kelvin's foam while losing to Weaire-Phelan.

This module relaxes the Frank-Kasper foams with the exp042 machinery: periodic
Voronoi diagram of the crystal sites, triangulated films, area minimised with
every cell at the same volume **and the periodic box free** (its shape is
optimised, not fixed at the crystal's lattice ratios), three refinement levels
and Richardson extrapolation. Coordinates are those of the prototype crystals
(cited per structure); the relaxation keeps their foam topology.

Structures: A15 (Weaire-Phelan), sigma, Z, mu, P, delta, the three Laves
polytypes C14, C15, C36, and alpha-Mn (chi; not Frank-Kasper: its CN13 sites
give 4-gons), with Kelvin's BCC as the reference. The R phase was tried and
left out: the Voronoi foam of its refined (disordered) coordinates, COD
1524030, has 21 quadrilateral faces, so it is not the Frank-Kasper foam.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp042_kelvin_problem import (
    SPHERE,
    foam_measures,
    periodic_voronoi,
    point_set,
    relax,
    triangulate,
)
from qres.bench import RESULTS_DIR

RAW_DIR = RESULTS_DIR / "raw" / "exp043"

# Each structure: cell (a, b, c, alpha, beta, gamma), space-group generators
# (International Tables setting of the cited file; the closure is checked
# against the Wyckoff multiplicities in the tests), asymmetric-unit sites with
# their multiplicities, the centring used to reduce to a primitive cell, and
# the source of the coordinates.
STRUCTURES = {
    "A15": {
        "cell": (4.556, 4.556, 4.556, 90, 90, 90), "group": "Pm-3n",
        "generators": ["z,x,y", "y+1/2,x+1/2,-z+1/2", "-x,-y,-z"],
        "sites": [((0, 0, 0), 2), ((0.25, 0, 0.5), 6)],
        "source": "Cr3Si; AFLOW A3B_cP8_223_c_a-001"},
    "sigma": {
        "cell": (8.7966, 8.7966, 4.5582, 90, 90, 90), "group": "P4_2/mnm",
        "generators": ["-y+1/2,x+1/2,z+1/2", "-x+1/2,y+1/2,-z+1/2", "-x,-y,-z"],
        "sites": [((0, 0, 0), 2), ((0.39864, 0.39864, 0), 4), ((0.46349, 0.13122, 0), 8),
                  ((0.73933, 0.06609, 0), 8), ((0.18267, 0.18267, 0.25202), 8)],
        "source": "CrFe (Yakel, Acta Cryst. B 39, 20 (1983)); AFLOW A4B4C4DE2_tP30_136_i_j_i_a_f-001"},
    "Z": {
        "cell": (5.433, 5.433, 5.39, 90, 90, 120), "group": "P6/mmm",
        "generators": ["-y,x-y,z", "-x,-y,z", "y,x,-z", "-x,-y,-z"],
        "sites": [((1 / 3, 2 / 3, 0.5), 2), ((0, 0, 0.25), 2), ((0.5, 0, 0), 3)],
        "source": "Zr4Al3 (Wilson et al., Acta Cryst. 13, 56 (1960)); AFLOW A3B4_hP7_191_f_de-001"},
    "mu": {
        "cell": (4.757, 4.757, 25.84, 90, 90, 120), "group": "R-3m", "centring": "R",
        "generators": ["-y,x-y,z", "y,x,-z", "-x,-y,-z", "x+2/3,y+1/3,z+1/3"],
        "sites": [((0, 0, 0), 3), ((0, 0, 0.167), 6), ((0, 0, 0.346), 6), ((0, 0, 0.448), 6),
                  ((0.5, 0.5, 0.59), 18)],
        "source": "W6Fe7; AFLOW A7B6_hR13_166_ah_3c-001"},
    "P": {
        "cell": (9.070, 16.983, 4.752, 90, 90, 90), "group": "Pbnm",
        "generators": ["x+1/2,-y+1/2,-z", "-x+1/2,y+1/2,-z+1/2", "-x,-y,-z"],
        "sites": [((0.3257, 0.1578, 0.25), 4), ((0.2504, 0.5375, 0.9986), 8),
                  ((0.3868, 0.2883, 0.0008), 8), ((0.9383, 0.365, 0.25), 4),
                  ((0.8152, 0.078, 0.25), 4), ((0.1363, 0.2547, 0.25), 4),
                  ((0.665, 0.3553, 0.25), 4), ((0.0737, 0.1134, 0.25), 4),
                  ((0.5202, 0.0355, 0.25), 4), ((0.6058, 0.1819, 0.25), 4),
                  ((0.4746, 0.4536, 0.25), 4), ((0.1988, 0.4047, 0.25), 4)],
        # COD 2310313 lists the edges as 16.983, 4.752, 9.07, which does not fit
        # its own coordinates (the Voronoi foam then has 9- to 20-face cells);
        # the paper's a = 9.070, b = 16.983, c = 4.752 gives the Frank-Kasper
        # foam, matching the a:b:c = 1.91:3.57:1 of Reddy et al.
        "source": "Mo-Ni-Cr P phase (Shoemaker, Shoemaker & Wilson, Acta Cryst. 10, 1 (1957)); COD 2310313"},
    "delta": {
        "cell": (9.108, 9.108, 8.852, 90, 90, 90), "group": "P2_12_12_1",
        "generators": ["-x+1/2,-y,z+1/2", "x+1/2,-y+1/2,-z"],
        "sites": [((0.1337, 0.0707, 0.2157), 4), ((0.1031, 0.4192, 0.9133), 4),
                  ((0.3186, 0.2464, 0.074), 4), ((0.068, 0.1442, 0.9529), 4),
                  ((0.2648, 0.1993, 0.7486), 4), ((0.3768, 0.4358, 0.8567), 4),
                  ((0.3882, 0.0523, 0.2748), 4), ((0.0029, 0.1969, 0.6767), 4),
                  ((0.2289, 0.2865, 0.4098), 4), ((0.1885, 0.0157, 0.496), 4),
                  ((0.1763, 0.4832, 0.6425), 4), ((0.0338, 0.3398, 0.1807), 4),
                  ((0.4424, 0.3662, 0.5971), 4), ((0.4519, 0.1153, 0.5322), 4)],
        "source": "MoNi delta phase (Shoemaker & Shoemaker, Acta Cryst. 16, 997 (1963)); COD 2310246"},
    "C15": {
        "cell": (7.02, 7.02, 7.02, 90, 90, 90), "group": "Fd-3m", "centring": "F",
        "generators": ["z,x,y", "y+3/4,x+1/4,-z+1/2", "-x,-y,-z"],
        "sites": [((0.375, 0.375, 0.375), 8), ((0, 0, 0), 16)],
        "source": "MgCu2; AFLOW A2B_cF24_227_c_b-001"},
    "C14": {
        "cell": (5.223, 5.223, 8.566, 90, 90, 120), "group": "P6_3/mmc",
        "generators": ["-y,x-y,z", "-x,-y,z+1/2", "y,x,-z", "-x,-y,-z"],
        "sites": [((0, 0, 0), 2), ((1 / 3, 2 / 3, 0.06286), 4), ((0.83048, 0.66097, 0.25), 6)],
        "source": "MgZn2; AFLOW AB2_hP12_194_f_ah-001"},
    "C36": {
        "cell": (4.824, 4.824, 15.826, 90, 90, 120), "group": "P6_3/mmc",
        "generators": ["-y,x-y,z", "-x,-y,z+1/2", "y,x,-z", "-x,-y,-z"],
        "sites": [((0, 0, 0.094), 4), ((1 / 3, 2 / 3, 0.84417), 4), ((1 / 3, 2 / 3, 0.12514), 4),
                  ((0.5, 0, 0), 6), ((0.16429, 0.32858, 0.25), 6)],
        "source": "MgNi2; AFLOW AB2_hP24_194_ef_fgh-001"},
    "chi": {
        "cell": (8.911, 8.911, 8.911, 90, 90, 90), "group": "I-43m", "centring": "I",
        "generators": ["-x,y,-z", "z,x,y", "y,x,z", "x+1/2,y+1/2,z+1/2"],
        "sites": [((0, 0, 0), 2), ((0.31787, 0.31787, 0.31787), 8),
                  ((0.08958, 0.08958, 0.28194), 24), ((0.35706, 0.35706, 0.03457), 24)],
        "source": "alpha-Mn; AFLOW A_cI58_217_ac2g-001"},
}

# Primitive vectors of the centred cells, in conventional fractional coordinates.
PRIMITIVE = {
    "F": ((0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
    "I": ((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)),
    "R": ((2 / 3, 1 / 3, 1 / 3), (-1 / 3, 1 / 3, 1 / 3), (-1 / 3, -2 / 3, 1 / 3)),
}

# Published dimensionless areas A = cost / (36 pi)^(1/3).
PUBLISHED = {
    "Kelvin": (1.097251, "Reddy et al. PNAS 2018"),
    "A15": (1.093541, "Reddy et al. PNAS 2018 (Weaire-Phelan)"),
    "P": (1.096795, "Reddy et al. PNAS 2018"),
    "sigma": (1.094759, "Phelan, as quoted by Reddy et al. (theirs is slightly higher)"),
}


# --------------------------------------------------------- crystallography

def parse_operation(text):
    """``"y+1/2,-x,z"`` -> (3x3 integer matrix, translation of Fractions)."""
    rotation = [[0, 0, 0] for _ in range(3)]
    shift = [Fraction(0)] * 3
    for row, part in enumerate(text.replace(" ", "").split(",")):
        for sign, term in _terms(part):
            if term in "xyz":
                rotation[row]["xyz".index(term)] += sign
            else:
                shift[row] += sign * Fraction(term)
    return tuple(map(tuple, rotation)), tuple(s % 1 for s in shift)


def _terms(part):
    out, k = [], 0
    while k < len(part):
        sign = 1
        if part[k] in "+-":
            sign = -1 if part[k] == "-" else 1
            k += 1
        end = k
        while end < len(part) and part[end] not in "+-":
            end += 1
        out.append((sign, part[k:end]))
        k = end
    return out


def _compose(first, second):
    """``first`` after ``second``, translations modulo the lattice."""
    (r1, t1), (r2, t2) = first, second
    rotation = tuple(tuple(sum(r1[i][k] * r2[k][j] for k in range(3)) for j in range(3))
                     for i in range(3))
    shift = tuple((sum(r1[i][k] * t2[k] for k in range(3)) + t1[i]) % 1 for i in range(3))
    return rotation, shift


def space_group(generators):
    """All operations generated, modulo lattice translations."""
    identity = (((1, 0, 0), (0, 1, 0), (0, 0, 1)), (Fraction(0),) * 3)
    gens = [parse_operation(g) for g in generators]
    group, frontier = {identity}, [identity]
    while frontier:
        fresh = []
        for g in frontier:
            for h in gens:
                c = _compose(g, h)
                if c not in group:
                    group.add(c)
                    fresh.append(c)
        frontier = fresh
    return sorted(group)


def orbit(site, operations, tolerance=2e-4):
    """Distinct images of a fractional site, wrapped into [0, 1)."""
    images = []
    for rotation, shift in operations:
        x = (np.array(rotation, float) @ np.array(site, float)
             + np.array([float(s) for s in shift])) % 1.0
        x[np.abs(x - 1) < tolerance] = 0.0
        if not any(np.all(np.abs((x - y + 0.5) % 1 - 0.5) < tolerance) for y in images):
            images.append(x)
    return images


def cell_matrix(a, b, c, alpha, beta, gamma):
    """Lattice vectors as rows, a along x, b in the xy plane."""
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    return np.array([[a, 0, 0], [b * math.cos(ga), b * math.sin(ga), 0],
                     [cx, cy, math.sqrt(c * c - cx * cx - cy * cy)]])


def build(name):
    """Unit cell (rows) and Cartesian sites of a structure, primitive cell,
    scaled to unit volume per site. Checks the Wyckoff multiplicities."""
    spec = STRUCTURES[name]
    operations = space_group(spec["generators"])
    frac = []
    for site, multiplicity in spec["sites"]:
        images = orbit(site, operations)
        if len(images) != multiplicity:
            raise ValueError(f"{name}: site {site} has {len(images)} images, "
                             f"expected {multiplicity}")
        frac += images
    box = cell_matrix(*spec["cell"])
    points = np.array(frac) @ box
    if spec.get("centring"):
        box = np.array(PRIMITIVE[spec["centring"]]) @ box
        inside = points @ np.linalg.inv(box)
        inside -= np.floor(inside + 1e-7)
        keep = []
        for x in inside:
            if not any(np.all(np.abs((x - y + 0.5) % 1 - 0.5) < 1e-5) for y in keep):
                keep.append(x)
        points = np.array(keep) @ box
    scale = (abs(np.linalg.det(box)) / len(points)) ** (-1 / 3)
    return box * scale, points * scale


# ------------------------------------------------------------------ foams

def census(foam):
    """Faces per cell (histogram), mean, and cells per vertex."""
    faces = np.zeros(len(foam["points"]), int)
    for face in foam["faces"]:
        a, b = face["cells"]
        faces[a] += 1
        faces[b] += 1
    measures = foam_measures(foam)
    volumes = measures["volumes"]
    return {"cells": len(faces),
            "faces_per_cell": {int(k): int(v) for k, v in zip(*np.unique(faces, return_counts=True))},
            "mean_faces": float(faces.mean()),
            "vertex_cells": {int(k): int(v) for k, v in measures["vertex_cells"].items()},
            "voronoi_volume_range": [float(volumes.min() / volumes.mean()),
                                     float(volumes.max() / volumes.mean())]}


def structure_foam(name):
    if name == "Kelvin":
        box, points = point_set("bcc")
    else:
        box, points = build(name)
    return periodic_voronoi(points, box)


def relax_level(name, level, raw_dir=RAW_DIR):
    """Free-box equal-volume relaxation of one structure at one refinement,
    cached in its own file so that parallel workers never share one."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}_level{level}.json"
    if path.exists():
        return json.loads(path.read_text())
    start = time.time()
    mesh = triangulate(structure_foam(name), level)
    result = relax(mesh, free_box=True)
    box = result["box"]
    # cost against the cells' actual mean volume (the box volume over the cell
    # count): with the box free, residual volume errors move the box volume,
    # and normalising by the target would carry that bias
    cells = mesh["cells"]
    mean_volume = abs(np.linalg.det(box)) / cells
    record = {"cost": 2 * result["area"] / cells / mean_volume ** (2 / 3),
              "cost_at_target_volume": result["cost"], "volume_error": result["volume_error"],
              "triangles": len(mesh["tri"]), "seconds": round(time.time() - start, 1),
              "box": box.tolist(), "box_lengths": np.linalg.norm(box, axis=1).tolist()}
    path.write_text(json.dumps(record, indent=1))
    print(f"  {name} level {level}: {record['cost']:.6f} "
          f"(volume error {result['volume_error']:.1e}, {record['seconds']:.0f} s)", flush=True)
    return record


def load_levels(name, raw_dir=RAW_DIR):
    """Every cached refinement of a structure, with its Voronoi census."""
    levels = {}
    for path in sorted(raw_dir.glob(f"{name}_level*.json")):
        levels[path.stem.rsplit("level", 1)[1]] = json.loads(path.read_text())
    return {"name": name, "census": census(structure_foam(name)), "levels": levels}


def relax_levels(name, levels=(1, 2, 3), raw_dir=RAW_DIR):
    """Relax at each requested level (missing ones are computed), then load."""
    for level in levels:
        relax_level(name, level, raw_dir)
    return load_levels(name, raw_dir)


def extrapolate(record):
    """Richardson extrapolation from the two finest levels, O(h^2)."""
    levels = sorted(int(k) for k in record["levels"])
    costs = [record["levels"][str(k)]["cost"] for k in levels]
    if len(costs) < 2:
        return costs[-1], None
    value = (4 * costs[-1] - costs[-2]) / 3
    return value, abs(value - costs[-1])


def summary(names, levels=(1, 2, 3), compute=True):
    """One row per structure. With ``compute`` False only the cache is read,
    and each structure is extrapolated from the finest levels it has."""
    rows = []
    for name in names:
        record = relax_levels(name, levels) if compute else load_levels(name)
        value, correction = extrapolate(record)
        finest = str(max(int(k) for k in record["levels"]))
        rows.append({"structure": name, "source": STRUCTURES.get(name, {}).get("source", "BCC"),
                     "census": record["census"],
                     "costs": {k: v["cost"] for k, v in record["levels"].items()},
                     "extrapolated": value, "last_correction": correction,
                     "A": value / SPHERE,
                     "published_A": PUBLISHED.get(name, (None,))[0],
                     "box_lengths": record["levels"][finest]["box_lengths"]})
    rows.sort(key=lambda row: row["extrapolated"])
    return rows


def selling_reduce(basis):
    """Selling reduction to an obtuse superbase: then every Voronoi-relevant
    vector is a 0/+-1 combination of the basis, so the 27 replicas used by
    ``periodic_voronoi`` see the whole Voronoi cell."""
    superbase = [-basis.sum(axis=0)] + list(basis)
    while True:
        pair = next(((i, j) for i, j in itertools.combinations(range(4), 2)
                     if superbase[i] @ superbase[j] > 1e-12), None)
        if pair is None:
            return np.array(superbase[1:])
        i, j = pair
        for k in range(4):
            if k not in (i, j):
                superbase[k] = superbase[k] + superbase[i]
        superbase[i] = -superbase[i]


def single_cell_scan(samples=24, level=2, seed=7, spread=0.4):
    """Kelvin among curved one-cell foams: the Voronoi foam of random lattices
    (one cell per lattice point), relaxed with the box free.

    Hales & Song prove the flat truncated octahedron optimal among
    parallelohedra; this asks the curved question numerically. Each start is a
    random lattice, ``I + spread * Gaussian``, Selling-reduced and scaled to
    unit volume; its Voronoi cell is generically of truncated-octahedral type
    (14 faces). Reports the relaxed costs, and whether the final lattice is
    body-centred cubic: the sorted lengths of its shortest vectors, a lattice
    invariant, against BCC's."""
    rng = np.random.default_rng(seed)
    primitive = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1]]) / 2.0
    reference = _shortest_vector_lengths(primitive / abs(np.linalg.det(primitive)) ** (1 / 3))
    out = []
    for _ in range(samples):
        basis = selling_reduce(np.eye(3) + spread * rng.normal(size=(3, 3)))
        basis /= abs(np.linalg.det(basis)) ** (1 / 3)
        foam = periodic_voronoi(np.zeros((1, 3)), basis)
        result = relax(triangulate(foam, level), free_box=True)
        box = selling_reduce(result["box"])
        cost = 2 * result["area"] / abs(np.linalg.det(box)) ** (2 / 3)
        lengths = _shortest_vector_lengths(box / abs(np.linalg.det(box)) ** (1 / 3))
        out.append({"faces": 2 * len(foam["faces"]), "cost": cost,
                    "volume_error": result["volume_error"],
                    "distance_to_bcc": float(np.max(np.abs(np.array(lengths) - reference)))})
    return {"samples": out, "cost_range": [min(r["cost"] for r in out), max(r["cost"] for r in out)],
            "max_distance_to_bcc": max(r["distance_to_bcc"] for r in out)}


def _shortest_vector_lengths(box, count=14):
    """Sorted lengths of the shortest nonzero lattice vectors: a lattice
    invariant, so two boxes describe the same lattice up to rotation exactly
    when these agree."""
    vectors = [np.array(c) @ box for c in itertools.product(range(-2, 3), repeat=3) if any(c)]
    return sorted(float(np.linalg.norm(v)) for v in vectors)[:count]


LABELS = {"A15": "A15", "sigma": "σ", "Z": "Z", "mu": "μ", "P": "P",
          "delta": "δ", "C15": "C15", "C14": "C14", "C36": "C36", "chi": "χ (α-Mn, not FK)"}
LABEL_OFFSETS = {"A15": (-4, 7), "Z": (7, 2)}           # clear of the reference lines


def survey_figure(rows, directory=RESULTS_DIR / "figures"):
    """Extrapolated cost against mean faces per cell, with Weaire-Phelan and
    Kelvin as reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    frank_kasper, other = "#2a78d6", "#eb6834"
    by_name = {r["structure"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    for name, value, text in (("A15", by_name["A15"]["extrapolated"], "Weaire-Phelan"),
                              ("Kelvin", by_name["Kelvin"]["extrapolated"], "Kelvin")):
        ax.axhline(value, color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(13.315, value, f"{text} {value:.5f}", color=muted, fontsize=8, va="bottom")
    laves = [n for n in ("C15", "C14", "C36") if n in by_name]
    for r in rows:
        name = r["structure"]
        if name == "Kelvin":
            continue
        x, y = r["census"]["mean_faces"], r["extrapolated"]
        colour = other if name == "chi" else frank_kasper
        ax.plot(x, y, "o", markersize=7, color=colour, markeredgecolor=surface,
                markeredgewidth=2, zorder=3)
        if name in laves[1:]:
            continue
        label = " = ".join(laves) if name == laves[0] else LABELS[name]
        ax.annotate(label, (x, y), xytext=LABEL_OFFSETS.get(name, (7, -3)),
                    textcoords="offset points", fontsize=8.5, color=ink)
    ax.set_xlabel("mean faces per cell", color=ink)
    ax.set_ylabel("cost = cell surface / volume$^{2/3}$ (relaxed, extrapolated)", color=ink)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
    ax.set_xlim(13.3, 13.56)
    ax.set_title("Frank-Kasper foams, equal volumes, periodic cell free: none beats Weaire-Phelan",
                 fontsize=10, color=ink, loc="left")
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "exp043_frank_kasper_costs.png"
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return path


NAMES = ["Kelvin", "A15", "sigma", "Z", "mu", "C15", "C14", "C36", "chi", "P", "delta"]


def main(argv=None) -> int:
    """``--only A,B --levels 1,2`` relaxes just those structures (a worker);
    ``--report-only`` summarises whatever is cached; no flags does everything."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    levels = tuple(int(x) for x in args.levels.split(","))
    if args.only:
        for name in args.only.split(","):
            for level in levels:
                relax_level(name, level)
        return 0
    names = NAMES
    rows = summary(names, levels, compute=not args.report_only)
    kelvin = next(r["extrapolated"] for r in rows if r["structure"] == "Kelvin")
    phelan = next(r["extrapolated"] for r in rows if r["structure"] == "A15")
    print(f"{'structure':10s} {'faces/cell':>10s} {'cost':>9s} {'A':>9s} {'vs WP':>8s} {'vs Kelvin':>9s}")
    for r in rows:
        print(f"{r['structure']:10s} {r['census']['mean_faces']:10.4f} {r['extrapolated']:9.5f} "
              f"{r['A']:9.6f} {100 * (r['extrapolated'] / phelan - 1):+7.3f}% "
              f"{100 * (r['extrapolated'] / kelvin - 1):+8.3f}%")
    scan_path = RAW_DIR / "single_cell_scan.json"           # 24 relaxations: cached like the levels
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
    else:
        scan = single_cell_scan()
        scan_path.parent.mkdir(parents=True, exist_ok=True)
        scan_path.write_text(json.dumps(scan))
    print(f"single-cell lattices: {len(scan['samples'])} random starts, relaxed costs "
          f"{scan['cost_range'][0]:.9f}-{scan['cost_range'][1]:.9f}, largest distance of the "
          f"final lattice from BCC {scan['max_distance_to_bcc']:.1e}")
    print(f"figure: {survey_figure(rows)}")
    path = RESULTS_DIR / "exp043_kelvin_search.json"
    path.write_text(json.dumps({"sphere": SPHERE, "rows": rows, "single_cell_scan": scan},
                               indent=1))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
