"""Experiment 042 -- the Kelvin problem: what is the 3-D analogue of the honeycomb?

A side programme, added at the request of the person who asked it on Math
StackExchange in 2015 (question 1494037, "What is the analog of the hexagonal
grid in 3-dimensional space? Rhombic dodecahedral honeycomb?").

**The problem.** Partition space into cells of equal volume with the least
interface area per cell. In 2-D the answer is the regular hexagonal honeycomb
(Hales 1999), which is also the Voronoi diagram of the densest circle packing.
The question asked whether 3-D works the same way, with the Voronoi diagram of
the densest sphere packing (FCC: rhombic dodecahedra; HCP: trapezo-rhombic
dodecahedra). It does not. Kelvin (1887) proposed slightly curved truncated
octahedra (the Voronoi cells of BCC); Weaire & Phelan (1993) beat Kelvin by
0.3% with two cell types (A15 structure). Whether Weaire-Phelan is optimal is
open. Hales & Song (arXiv:2609.00997, 2026) prove the restricted statement
that among *parallelohedra* -- one flat-faced cell tiling by translations --
the truncated octahedron is optimal.

**Cost.** For equal-volume cells, ``cost = (surface of a cell) / V^(2/3)``
averaged over cells, each interface counted once for each cell it bounds. The
sphere's ``(36 pi)^(1/3) = 4.836`` is the trivial lower bound.

**What this module does.**

1. *Exact costs* of the flat candidates with SymPy: cube, optimal hexagonal
   prism, rhombic dodecahedron (FCC), trapezo-rhombic dodecahedron (HCP),
   truncated octahedron (BCC). Each Voronoi cell is computed exactly as an
   intersection of half-spaces.
2. *Periodic Voronoi foams* of point sets (SC, BCC, FCC, HCP, A15), with the
   vertex census that decides Plateau's laws: in an area-minimizing foam,
   exactly four cells meet at every vertex and three faces at 120 degrees along
   every edge (J. Taylor, Ann. Math. 1976). A structure with a vertex where six
   cells meet can be locally improved, so it is not optimal.
3. *Relaxation* of a periodic foam to curved soap films with equal cell
   volumes (``relax``): triangulated faces, area minimised under exact volume
   constraints by projected gradient descent -- a small stand-in for Brakke's
   Surface Evolver, checked against the published Kelvin and Weaire-Phelan
   costs.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import RESULTS_DIR

SPHERE = (36 * math.pi) ** (1 / 3)


# --------------------------------------------------------------- exact cells

def exact_voronoi_cell(neighbours):
    """Exact Voronoi cell of the origin among ``neighbours`` (SymPy vectors).

    The cell is ``{x : x.n <= |n|^2 / 2 for every neighbour n}``. Vertices are
    the feasible intersections of three bisector planes; faces are grouped by
    plane. Returns ``(vertices, faces)`` with faces as vertex-index lists in
    cyclic order (outward normal by the right-hand rule).
    """
    import sympy as sp

    planes = [(sp.Matrix(n), sp.Matrix(n).dot(sp.Matrix(n)) / 2) for n in neighbours]
    vertices = []
    for (n1, c1), (n2, c2), (n3, c3) in itertools.combinations(planes, 3):
        matrix = sp.Matrix.hstack(n1, n2, n3).T
        if sp.simplify(matrix.det()) == 0:
            continue
        point = sp.simplify(matrix.LUsolve(sp.Matrix([c1, c2, c3])))
        if (all(sp.simplify(point.dot(n) - c) <= 0 for n, c in planes)
                and not any(sp.simplify((point - v).norm()) == 0 for v in vertices)):
            vertices.append(point)
    faces = []
    for n, c in planes:
        on = [k for k, v in enumerate(vertices) if sp.simplify(v.dot(n) - c) == 0]
        if len(on) < 3:
            continue
        centre = sum((vertices[k] for k in on), sp.zeros(3, 1)) / len(on)
        normal = n / n.norm()
        # order by angle in the face plane
        axis = (vertices[on[0]] - centre)
        axis = axis / axis.norm()
        other = normal.cross(axis)

        def angle(k, centre=centre, axis=axis, other=other):
            d = vertices[k] - centre
            return math.atan2(float(d.dot(other)), float(d.dot(axis)))

        faces.append(sorted(on, key=angle))
    return vertices, faces


def exact_area_volume(vertices, faces):
    """Exact surface area and volume of a convex polytope given by faces."""
    import sympy as sp

    area = sp.Integer(0)
    volume = sp.Integer(0)
    for face in faces:
        p0 = vertices[face[0]]
        for a, b in itertools.pairwise(face[1:]):
            cross = (vertices[a] - p0).cross(vertices[b] - p0)
            area += sp.sqrt(sp.simplify(cross.dot(cross))) / 2
            volume += p0.dot((vertices[a]).cross(vertices[b])) / 6
    return sp.nsimplify(sp.simplify(area)), sp.simplify(volume)


def lattice_shells(basis, radius):
    """All nonzero lattice vectors of norm <= radius (exact SymPy)."""
    import sympy as sp

    basis = [sp.Matrix(b) for b in basis]
    out = []
    reach = 3
    for coefficients in itertools.product(range(-reach, reach + 1), repeat=3):
        if coefficients == (0, 0, 0):
            continue
        v = sum((c * b for c, b in zip(coefficients, basis)), sp.zeros(3, 1))
        if float(v.norm()) <= radius + 1e-9:
            out.append(v)
    return out


def exact_candidates():
    """Exact cost of each classical flat candidate, ``A / V^(2/3)``."""
    import sympy as sp

    s2, s3 = sp.sqrt(2), sp.sqrt(3)
    results = {}

    # cube
    results["cube (simple cubic Voronoi)"] = sp.Integer(6)

    # best hexagonal prism: side a, height h, minimise at fixed volume
    a, h = sp.symbols("a h", positive=True)
    area = 3 * s3 * a ** 2 + 6 * a * h
    volume = 3 * s3 / 2 * a ** 2 * h
    best_h = sp.solve(sp.diff((area ** 3 / volume ** 2).subs(a, 1), h), h)[0]
    results["hexagonal prism (best height)"] = sp.simplify(
        (area / volume ** sp.Rational(2, 3)).subs({a: 1, h: best_h}))

    # Voronoi cells of the three sphere packings and of BCC
    fcc = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    bcc = [(1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    for name, basis, radius in (
        ("rhombic dodecahedron (FCC, densest packing)", fcc, sp.sqrt(2)),
        ("truncated octahedron (BCC, Kelvin's cell flat)", bcc, 2),
    ):
        cell = exact_voronoi_cell(lattice_shells(basis, float(radius)))
        A, V = exact_area_volume(*cell)
        results[name] = sp.simplify(A / V ** sp.Rational(2, 3))

    # HCP: the 12 nearest neighbours of a site (unit spacing)
    hexagon = [(sp.cos(k * sp.pi / 3), sp.sin(k * sp.pi / 3), 0) for k in range(6)]
    up = [(sp.Rational(1, 2), s3 / 6, s2 / s3), (-sp.Rational(1, 2), s3 / 6, s2 / s3),
          (0, -s3 / 3, s2 / s3)]
    down = [(x, y, -z) for x, y, z in up]
    cell = exact_voronoi_cell([sp.Matrix(v) for v in hexagon + up + down])
    A, V = exact_area_volume(*cell)
    results["trapezo-rhombic dodecahedron (HCP, densest packing)"] = sp.simplify(
        A / V ** sp.Rational(2, 3))
    return results


# ------------------------------------------------------ periodic point sets

def point_set(name):
    """Unit cell (rows = lattice vectors) and points, for the named structure."""
    if name == "sc":
        return np.eye(3), np.zeros((1, 3))
    if name == "bcc":
        return np.eye(3), np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    if name == "fcc":
        return np.eye(3), np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    if name == "hcp":
        box = np.array([[1, 0, 0], [0.5, math.sqrt(3) / 2, 0], [0, 0, math.sqrt(8 / 3)]])
        frac = np.array([[0, 0, 0], [1 / 3, 1 / 3, 0.5]])
        return box, frac @ box
    if name == "a15":   # Weaire-Phelan: the centres in Brakke's phelanc.fe, cube of side 2
        pts = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0, 1], [1.5, 0, 1], [0, 1, 0.5],
                        [0, 1, 1.5], [1, 0.5, 0], [1, 1.5, 0]]) / 2
        return np.eye(3), pts
    raise KeyError(name)


# ------------------------------------------------------- periodic Voronoi

def periodic_voronoi(points, box, decimals=7):
    """Periodic Voronoi foam of ``points`` in the cell spanned by ``box`` rows.

    Returns a dict with canonical vertices (wrapped into the cell), faces, and
    per-face data: the two cells, the image shift of the second, and the
    polygon as ``(vertex id, integer shift)`` pairs expressed in the frame of
    the first cell, ordered so the normal points from the first cell to the
    second.
    """
    from scipy.spatial import Voronoi

    n = len(points)
    inverse = np.linalg.inv(box)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    centre = int(np.where((shifts == 0).all(axis=1))[0][0])
    replicated = (points[None, :, :] + (shifts @ box)[:, None, :]).reshape(-1, 3)
    vor = Voronoi(replicated)

    keys, vertex_positions = {}, []

    def canonical(x):
        frac = x @ inverse
        wrapped = frac - np.floor(np.round(frac, decimals))
        key = tuple(np.round(wrapped, decimals) % 1.0)
        key = tuple(round(v, decimals - 1) for v in key)
        if key not in keys:
            keys[key] = len(vertex_positions)
            vertex_positions.append(wrapped @ box)
        vid = keys[key]
        shift = np.rint(frac - wrapped).astype(int)
        return vid, tuple(int(s) for s in shift)

    faces, seen = [], set()
    for (p, q), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if p // n != centre and q // n != centre:
            continue
        if p // n != centre:
            p, q = q, p
        a, b = p % n, q % n
        shift = tuple(int(s) for s in shifts[q // n] - shifts[p // n])
        key = (a, b, shift)
        mirror = (b, a, tuple(-s for s in shift))
        if key in seen or mirror in seen:
            continue
        seen.add(key)
        if -1 in ridge:
            raise RuntimeError("unbounded ridge next to a central cell")
        corners = vor.vertices[ridge]
        normal = replicated[q] - replicated[p]
        middle = corners.mean(axis=0)
        u = corners[0] - middle
        u /= np.linalg.norm(u)
        w = np.cross(normal / np.linalg.norm(normal), u)
        order = np.argsort(np.arctan2((corners - middle) @ w, (corners - middle) @ u))
        polygon = [canonical(corners[k]) for k in order]
        faces.append({"cells": (int(a), int(b)), "shift": shift, "polygon": polygon,
                      "points": corners[order]})
    return {"box": box, "points": points, "vertices": np.array(vertex_positions),
            "faces": faces}


def foam_measures(foam):
    """Total interface area, cell volumes and the Plateau census of a flat foam."""
    n = len(foam["points"])
    volumes = np.zeros(n)
    area = 0.0
    incident = {}
    for face in foam["faces"]:
        pts = face["points"]
        a, b = face["cells"]
        shift = np.array(face["shift"]) @ foam["box"]
        face_area = 0.0
        for k in range(1, len(pts) - 1):
            cross = np.cross(pts[k] - pts[0], pts[k + 1] - pts[0])
            face_area += 0.5 * np.linalg.norm(cross)
            # divergence theorem: outward for a, inward for b (in b's frame)
            volumes[a] += np.dot(pts[0], np.cross(pts[k], pts[k + 1])) / 6
            volumes[b] -= np.dot(pts[0] - shift, np.cross(pts[k] - shift, pts[k + 1] - shift)) / 6
        area += face_area
        for vid, vshift in face["polygon"]:
            incident.setdefault(vid, set())
            # the cells around the vertex, as (cell, cell image relative to vertex)
            incident[vid].add((a, tuple(-np.array(vshift))))
            incident[vid].add((b, tuple(np.array(face["shift"]) - np.array(vshift))))
    census = {}
    for cells in incident.values():
        census[len(cells)] = census.get(len(cells), 0) + 1
    return {"area": area, "volumes": volumes, "vertex_cells": census}


def flat_cost(foam):
    measures = foam_measures(foam)
    mean_volume = measures["volumes"].mean()
    n = len(foam["points"])
    return 2 * measures["area"] / n / mean_volume ** (2 / 3), measures


def edge_angles(foam):
    """Angles between the faces meeting along each edge of a flat foam (degrees).

    For every edge, each incident face contributes the unit vector in its plane
    perpendicular to the edge and pointing into the face. Plateau's laws ask for
    exactly three faces per edge, 120 degrees apart. Returns, per edge, the
    number of faces and the sorted pairwise angles.
    """
    edges = {}
    for face in foam["faces"]:
        pts = face["points"]
        centroid = pts.mean(axis=0)
        polygon = face["polygon"]
        for k in range(len(polygon)):
            (u, su), (v, sv) = polygon[k], polygon[(k + 1) % len(polygon)]
            su, sv = np.array(su), np.array(sv)
            if (u, tuple(su)) > (v, tuple(sv)):
                (u, su), (v, sv) = (v, sv), (u, su)
            key = (u, v, tuple(sv - su))
            a, b = pts[k], pts[(k + 1) % len(pts)]
            direction = (b - a) / np.linalg.norm(b - a)
            inward = centroid - 0.5 * (a + b)
            inward -= np.dot(inward, direction) * direction
            edges.setdefault(key, []).append(inward / np.linalg.norm(inward))
    out = []
    for vectors in edges.values():
        angles = sorted(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(x, y))))))
                        for x, y in itertools.combinations(vectors, 2))
        out.append((len(vectors), angles))
    return out


# ------------------------------------------------------- soap-film relaxation

def _edge_key(u, su, v, sv):
    """Frame-independent key of the edge (u at su) -- (v at sv); returns the key
    and which endpoint (0 or 1) the midpoint is stored relative to."""
    d = tuple(int(x) for x in np.subtract(sv, su))
    if u < v or (u == v and d > (0, 0, 0)):
        return (u, v, d), 0
    return (v, u, tuple(-x for x in d)), 1


def _midpoint(y, box, midpoints, u, su, v, sv):
    """Shared midpoint of an edge, created once; returns (id, its image shift)."""
    key, first = _edge_key(u, tuple(su), v, tuple(sv))
    base = su if first == 0 else sv
    if key not in midpoints:
        pu = y[u] + np.array(su) @ box
        pv = y[v] + np.array(sv) @ box
        midpoints[key] = len(y)
        y.append(0.5 * (pu + pv) - np.array(base) @ box)
    return midpoints[key], base


def triangulate(foam, refine=0):
    """Triangulated periodic foam: centroid fans, then ``refine`` 1:4 splits.

    Returns a dict with positions ``y`` (mesh vertices, unwrapped), triangles
    ``tri`` (vertex ids), per-corner integer image shifts ``shift`` (position of
    corner k is ``y[tri[t,k]] + shift[t,k] @ box``, in the frame of the first
    cell of the face), the two cells of each triangle and the image shift of the
    second cell. Triangle normals point from the first cell to the second.
    """
    box = foam["box"]
    y = [np.array(v, float) for v in foam["vertices"]]
    tri, shift, cell_a, cell_b, cell_shift = [], [], [], [], []
    for face in foam["faces"]:
        corners = face["polygon"]
        points = face["points"]
        centre = len(y)
        y.append(points.mean(axis=0))
        for k in range(len(corners)):
            (u, su), (v, sv) = corners[k], corners[(k + 1) % len(corners)]
            tri.append((centre, u, v))
            shift.append(((0, 0, 0), su, sv))
            cell_a.append(face["cells"][0])
            cell_b.append(face["cells"][1])
            cell_shift.append(face["shift"])
    tri, shift = np.array(tri), np.array(shift)
    for _ in range(refine):
        midpoints = {}
        new_tri, new_shift = [], []
        for t in range(len(tri)):
            (i, j, k), (si, sj, sk) = tri[t], shift[t]
            ij, sij = _midpoint(y, box, midpoints, i, si, j, sj)
            jk, sjk = _midpoint(y, box, midpoints, j, sj, k, sk)
            ki, ski = _midpoint(y, box, midpoints, k, sk, i, si)
            new_tri += [(i, ij, ki), (ij, j, jk), (ki, jk, k), (ij, jk, ki)]
            new_shift += [(si, sij, ski), (sij, sj, sjk), (ski, sjk, sk), (sij, sjk, ski)]
        parents = np.repeat(np.arange(len(tri)), 4)
        cell_a = [cell_a[t] for t in parents]
        cell_b = [cell_b[t] for t in parents]
        cell_shift = [cell_shift[t] for t in parents]
        tri, shift = np.array(new_tri), np.array(new_shift)
    return {"box": box, "y": np.array(y), "tri": tri, "shift": shift,
            "cell_a": np.array(cell_a), "cell_b": np.array(cell_b),
            "cell_shift": np.array(cell_shift), "cells": len(foam["points"])}


def surface_terms(y, mesh):
    """Total area, cell volumes and their gradients with respect to ``y``."""
    box, tri = mesh["box"], mesh["tri"]
    p = y[tri] + mesh["shift"] @ box
    p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
    normal = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normal, axis=1)
    area = 0.5 * length.sum()
    unit = normal / length[:, None]
    grad_area = np.zeros_like(y)
    for k, (a, b) in enumerate(((p1, p2), (p2, p0), (p0, p1))):
        np.add.at(grad_area, tri[:, k], 0.5 * np.cross(a - b, unit))
    cells = mesh["cells"]
    volumes = np.zeros(cells)
    grad_volume = np.zeros((cells,) + y.shape)
    offset = mesh["cell_shift"] @ box
    for sign, cell, q in ((1.0, mesh["cell_a"], p), (-1.0, mesh["cell_b"], p - offset[:, None, :])):
        q0, q1, q2 = q[:, 0], q[:, 1], q[:, 2]
        np.add.at(volumes, cell, sign * np.einsum("ij,ij->i", q0, np.cross(q1, q2)) / 6)
        for k, (a, b) in enumerate(((q1, q2), (q2, q0), (q0, q1))):
            np.add.at(grad_volume, (cell, tri[:, k]), sign * np.cross(a, b) / 6)
    return area, volumes, grad_area, grad_volume


def relax(mesh, target=None, rho=50.0, tolerance=1e-10, rounds=40, inner=400):
    """Minimise total area with every cell at the same volume.

    Augmented Lagrangian: ``A - lam.(V - V0) + rho/2 |V - V0|^2`` minimised by
    L-BFGS, multipliers updated between rounds. The multipliers converge to
    the cells' pressures. Returns the relaxed positions, the cost per cell
    ``2 A / n / V0^(2/3)`` and diagnostics.
    """
    from scipy.optimize import minimize

    cells = mesh["cells"]
    target = abs(np.linalg.det(mesh["box"])) / cells if target is None else target
    lam = np.zeros(cells)
    y = mesh["y"].copy()
    shape = y.shape

    def objective(flat):
        area, volumes, ga, gv = surface_terms(flat.reshape(shape), mesh)
        gap = volumes - target
        value = area - lam @ gap + 0.5 * rho * gap @ gap
        grad = ga - np.einsum("c,cij->ij", lam - rho * gap, gv)
        return value, grad.ravel()

    for _ in range(rounds):
        result = minimize(objective, y.ravel(), jac=True, method="L-BFGS-B",
                          options={"maxiter": inner, "ftol": 1e-15, "gtol": 1e-11})
        y = result.x.reshape(shape)
        area, volumes, _, _ = surface_terms(y, mesh)
        gap = volumes - target
        lam = lam - rho * gap
        if np.abs(gap).max() < tolerance * target and result.nit < inner:
            break
    area, volumes, _, _ = surface_terms(y, mesh)
    return {"y": y, "area": area, "cost": 2 * area / cells / target ** (2 / 3),
            "volume_error": float(np.abs(volumes / target - 1).max()),
            "pressures": lam.tolist()}


def relaxed_cost(structure, levels=(1, 2, 3)):
    """Relaxed cost at several refinements and its Richardson extrapolation.

    Each refinement halves the mesh size, and the discretisation error of a
    triangulated smooth surface is O(h^2), so ``(4 c_fine - c_coarse) / 3``.
    """
    box, points = point_set(structure)
    foam = periodic_voronoi(points, box)
    out = {}
    for level in levels:
        result = relax(triangulate(foam, level))
        out[level] = result
    costs = [out[level]["cost"] for level in levels]
    extrapolated = (4 * costs[-1] - costs[-2]) / 3 if len(costs) > 1 else costs[-1]
    return {"levels": list(levels), "costs": costs, "extrapolated": extrapolated,
            "volume_error": max(out[level]["volume_error"] for level in levels)}


def main() -> int:
    print("=== experiment 042 :: the Kelvin problem ===\n")
    print("Exact costs of the flat candidates, A / V^(2/3) (sphere bound 4.8360):")
    exact = exact_candidates()
    rows = []
    for name, value in sorted(exact.items(), key=lambda kv: float(kv[1])):
        print(f"  {float(value):.6f}   {name}\n             = {value}")
        rows.append({"structure": name, "cost": float(value), "exact": str(value)})
    print("\nPeriodic Voronoi foams: flat cost and how many cells meet at each vertex")
    foams = []
    for name in ("sc", "bcc", "fcc", "hcp", "a15"):
        box, pts = point_set(name)
        foam = periodic_voronoi(pts, box)
        cost, measures = flat_cost(foam)
        spread = measures["volumes"].max() / measures["volumes"].min() - 1
        census = dict(sorted(measures["vertex_cells"].items()))
        angles = edge_angles(foam)
        faces_per_edge = sorted({k for k, _ in angles})
        worst = max(abs(a - 120.0) for k, angs in angles if k == 3 for a in angs) \
            if any(k == 3 for k, _ in angles) else float("nan")
        print(f"  {name:>4}: cost {cost:.6f}  volume spread {spread:.4f}  "
              f"cells per vertex {census}  faces per edge {faces_per_edge}  "
              f"worst |angle - 120| {worst:.2f} deg")
        foams.append({"structure": name, "flat_cost": cost, "volume_spread": spread,
                      "cells_per_vertex": {str(k): v for k, v in census.items()},
                      "faces_per_edge": faces_per_edge, "worst_angle_deviation": worst})
    path = RESULTS_DIR / "exp042_kelvin_problem.json"
    path.write_text(json.dumps({"sphere_bound": SPHERE, "exact": rows, "voronoi": foams},
                               indent=1))
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
