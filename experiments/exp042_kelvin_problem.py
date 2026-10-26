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
    tri, shift, cell_a, cell_b, cell_shift, face_of = [], [], [], [], [], []
    for number, face in enumerate(foam["faces"]):
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
            face_of.append(number)
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
        face_of = [face_of[t] for t in parents]
        tri, shift = np.array(new_tri), np.array(new_shift)
    return {"box": box, "y": np.array(y), "tri": tri, "shift": shift,
            "cell_a": np.array(cell_a), "cell_b": np.array(cell_b),
            "cell_shift": np.array(cell_shift), "face": np.array(face_of),
            "cells": len(foam["points"])}


def surface_terms(y, mesh, box=None, want_box=False):
    """Total area, cell volumes and their gradients with respect to ``y``.

    With ``want_box`` also the derivatives with respect to the periodic box at
    fixed ``y`` through the image shifts only (``sum_corners shift^T grad``),
    which is what the chain rule needs when the box is a variable.
    """
    box = mesh["box"] if box is None else box
    tri = mesh["tri"]
    p = y[tri] + mesh["shift"] @ box
    p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
    normal = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normal, axis=1)
    area = 0.5 * length.sum()
    unit = normal / length[:, None]
    grad_area = np.zeros_like(y)
    box_area = np.zeros((3, 3))
    for k, (a, b) in enumerate(((p1, p2), (p2, p0), (p0, p1))):
        corner = 0.5 * np.cross(a - b, unit)
        np.add.at(grad_area, tri[:, k], corner)
        if want_box:
            box_area += mesh["shift"][:, k].T @ corner
    cells = mesh["cells"]
    volumes = np.zeros(cells)
    grad_volume = np.zeros((cells,) + y.shape)
    box_volume = np.zeros((cells, 3, 3))
    offset = mesh["cell_shift"] @ box
    passes = ((1.0, mesh["cell_a"], p, mesh["shift"]),
              (-1.0, mesh["cell_b"], p - offset[:, None, :],
               mesh["shift"] - mesh["cell_shift"][:, None, :]))
    for sign, cell, q, shifts in passes:
        q0, q1, q2 = q[:, 0], q[:, 1], q[:, 2]
        np.add.at(volumes, cell, sign * np.einsum("ij,ij->i", q0, np.cross(q1, q2)) / 6)
        for k, (a, b) in enumerate(((q1, q2), (q2, q0), (q0, q1))):
            corner = sign * np.cross(a, b) / 6
            np.add.at(grad_volume, (cell, tri[:, k]), corner)
            if want_box:
                np.add.at(box_volume, cell, np.einsum("ti,tj->tij", shifts[:, k], corner))
    if want_box:
        return area, volumes, grad_area, grad_volume, box_area, box_volume
    return area, volumes, grad_area, grad_volume


def relax(mesh, target=None, rho=50.0, tolerance=1e-10, rounds=40, inner=400,
          free_box=False):
    """Minimise total area with every cell at the same volume.

    Augmented Lagrangian: ``A - lam.(V - V0) + rho/2 |V - V0|^2`` minimised by
    L-BFGS, multipliers updated between rounds. The multipliers converge to
    the cells' pressures. With ``free_box`` the periodic box is a variable too
    (vertices are carried in fractional coordinates); the volume constraints
    keep its volume fixed. Returns the relaxed positions, the box, the cost
    per cell ``2 A / n / V0^(2/3)`` and diagnostics.
    """
    from scipy.optimize import minimize

    cells = mesh["cells"]
    box0 = np.array(mesh["box"], float)
    target = abs(np.linalg.det(box0)) / cells if target is None else target
    lam = np.zeros(cells)
    shape = mesh["y"].shape
    frac0 = mesh["y"] @ np.linalg.inv(box0)

    def unpack(x):
        if not free_box:
            return x.reshape(shape), box0
        frac = x[:-9].reshape(shape)
        box = x[-9:].reshape(3, 3)
        return frac @ box, box

    def objective(x):
        y, box = unpack(x)
        if not free_box:
            area, volumes, ga, gv = surface_terms(y, mesh)
            gap = volumes - target
            weights = lam - rho * gap
            grad = ga - np.einsum("c,cij->ij", weights, gv)
            return area - lam @ gap + 0.5 * rho * gap @ gap, grad.ravel()
        area, volumes, ga, gv, ba, bv = surface_terms(y, mesh, box, want_box=True)
        gap = volumes - target
        weights = lam - rho * gap
        grad_y = ga - np.einsum("c,cij->ij", weights, gv)
        grad_box_shift = ba - np.einsum("c,cij->ij", weights, bv)
        frac = x[:-9].reshape(shape)
        grad_frac = grad_y @ box.T
        grad_box = frac.T @ grad_y + grad_box_shift
        value = area - lam @ gap + 0.5 * rho * gap @ gap
        return value, np.concatenate([grad_frac.ravel(), grad_box.ravel()])

    x = (np.concatenate([frac0.ravel(), box0.ravel()]) if free_box
         else mesh["y"].ravel().copy())
    for _ in range(rounds):
        result = minimize(objective, x, jac=True, method="L-BFGS-B",
                          options={"maxiter": inner, "ftol": 1e-15, "gtol": 1e-11})
        x = result.x
        y, box = unpack(x)
        area, volumes = surface_terms(y, mesh, box)[:2]
        gap = volumes - target
        lam = lam - rho * gap
        if np.abs(gap).max() < tolerance * target and result.nit < inner:
            break
    y, box = unpack(x)
    area, volumes = surface_terms(y, mesh, box)[:2]
    return {"y": y, "box": box, "area": area, "cost": 2 * area / cells / target ** (2 / 3),
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


# ------------------------------------------- opening the six-cell vertices

def six_cell_vertices(foam):
    """Vertices where six cells meet, with those cells as (cell, image) pairs
    relative to the vertex's canonical position."""
    around = {}
    for face in foam["faces"]:
        a, b = face["cells"]
        for vid, vshift in face["polygon"]:
            around.setdefault(vid, set())
            around[vid].add((a, tuple(-np.array(vshift))))
            around[vid].add((b, tuple(np.array(face["shift"]) - np.array(vshift))))
    return {vid: sorted(cells) for vid, cells in around.items() if len(cells) == 6}


def open_six_cell_vertices(foam, choose, size=0.03):
    """Replace every six-cell vertex by a small square face, as a soap film does
    in a cubic wire frame.

    The six cells around such a vertex sit in three opposite pairs (the
    octahedral hole of the packing). ``choose(vid, pairs)`` picks the pair that
    becomes the square's two sides; the other four cells keep meeting along the
    square's edges. Every new vertex has four cells around it, as Plateau's
    laws require. Returns a new foam.
    """
    box, points = foam["box"], foam["points"]
    vertices = [np.array(v) for v in foam["vertices"]]
    faces = [dict(face, polygon=list(face["polygon"])) for face in foam["faces"]]
    replace = {}   # vid -> data for rewriting faces
    for vid, cells in six_cell_vertices(foam).items():
        centre = vertices[vid]
        direction = {c: points[c[0]] + np.array(c[1]) @ box - centre for c in cells}
        unit = {c: d / np.linalg.norm(d) for c, d in direction.items()}
        pairs = []
        for c1, c2 in itertools.combinations(cells, 2):
            if np.dot(unit[c1], unit[c2]) < -0.9:
                pairs.append((c1, c2))
        if len(pairs) != 3:
            raise RuntimeError(f"vertex {vid} is not an octahedral six-cell vertex")
        top, bottom = pairs[choose(vid, pairs)]
        axis = unit[top] - unit[bottom]
        axis /= np.linalg.norm(axis)
        sides = [c for c in cells if c not in (top, bottom)]
        corners = {}
        for x, y in itertools.combinations(sides, 2):
            if np.dot(unit[x], unit[y]) < -0.9:
                continue   # opposite sides do not share a corner
            out = unit[x] + unit[y]
            out -= np.dot(out, axis) * axis
            corners[frozenset((x, y))] = len(vertices)
            vertices.append(centre + size * out / np.linalg.norm(out))
        replace[vid] = {"top": top, "bottom": bottom, "sides": sides, "corners": corners,
                        "axis": axis}
    for face in faces:
        a, b = face["cells"]
        polygon = []
        for k, (vid, vshift) in enumerate(face["polygon"]):
            if vid not in replace:
                polygon.append((vid, vshift))
                continue
            data = replace[vid]
            here = {(a, tuple(-np.array(vshift))),
                    (b, tuple(np.array(face["shift"]) - np.array(vshift)))}
            ends = here & {data["top"], data["bottom"]}
            if ends:
                side = next(iter(here - ends))
                new = [cid for pair, cid in data["corners"].items() if side in pair]
            else:
                new = [data["corners"][frozenset(here)]]
            if len(new) == 2:   # keep the polygon simple: nearer the previous vertex first
                previous_vid, previous_shift = face["polygon"][k - 1]
                previous = vertices[previous_vid] + np.array(previous_shift) @ box
                new.sort(key=lambda cid, vs=vshift: np.linalg.norm(
                    vertices[cid] + np.array(vs) @ box - previous))
            polygon += [(cid, vshift) for cid in new]
        face["polygon"] = polygon
    for vid, data in replace.items():
        (top_cell, top_image), (bottom_cell, bottom_image) = data["top"], data["bottom"]
        frame = -np.array(top_image)
        ring = list(data["corners"].values())
        centre = np.mean([vertices[c] for c in ring], axis=0)
        u = vertices[ring[0]] - centre
        u /= np.linalg.norm(u)
        normal = -data["axis"]          # from top to bottom
        w = np.cross(normal, u)
        ring.sort(key=lambda c: math.atan2(np.dot(vertices[c] - centre, w),
                                           np.dot(vertices[c] - centre, u)))
        faces.append({"cells": (top_cell, bottom_cell),
                      "shift": tuple(int(v) for v in np.array(bottom_image) - np.array(top_image)),
                      "polygon": [(c, tuple(int(v) for v in frame)) for c in ring]})
    for face in faces:
        face["points"] = np.array([vertices[vid] + np.array(sh) @ box
                                   for vid, sh in face["polygon"]])
    return {"box": box, "points": points, "vertices": np.array(vertices), "faces": faces}


# --------------------------------------------------------------- pictures

def cell_polygons(foam, cell):
    """The faces of one cell of a flat foam, in that cell's frame."""
    out = []
    for face in foam["faces"]:
        a, b = face["cells"]
        offset = np.array(face["shift"]) @ foam["box"]
        if a == cell:
            out.append(face["points"])
        if b == cell:
            out.append(face["points"][::-1] - offset)
    return out


def cell_triangles(mesh, y, cell):
    """The triangles bounding one cell of a relaxed mesh, in that cell's frame,
    and the curved foam edges (segments where two of the cell's faces meet)."""
    p = y[mesh["tri"]] + mesh["shift"] @ mesh["box"]
    offset = mesh["cell_shift"] @ mesh["box"]
    as_a, as_b = mesh["cell_a"] == cell, mesh["cell_b"] == cell
    triangles = np.concatenate([p[as_a], (p - offset[:, None, :])[as_b][:, ::-1]])
    faces = np.concatenate([mesh["face"][as_a], mesh["face"][as_b]])
    owner = {}
    for t, corners in enumerate(triangles):
        for k in range(3):
            a, b = corners[k], corners[(k + 1) % 3]
            key = tuple(sorted((tuple(np.round(a, 6)), tuple(np.round(b, 6)))))
            owner.setdefault(key, set()).add(int(faces[t]))
    edges = [np.array(key) for key, f in owner.items() if len(f) > 1]
    return triangles, edges


def draw(ax, polygons, colour, title, edges=True, lines=(), view=(22, 35)):
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    centre = np.concatenate(polygons).mean(axis=0)
    shifted = [poly - centre for poly in polygons]
    collection = Poly3DCollection(shifted, facecolors=colour, alpha=0.95, shade=True,
                                  lightsource=LightSource(azdeg=315, altdeg=35),
                                  edgecolor="#1b1b1b" if edges else "none",
                                  linewidths=0.6 if edges else 0.0)
    ax.add_collection3d(collection)
    if len(lines):
        # only the edges on the side facing the camera, drawn on top
        elevation, azimuth = np.radians(view)
        eye = np.array([np.cos(elevation) * np.cos(azimuth),
                        np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        front = [seg - centre for seg in lines if np.dot(seg.mean(axis=0) - centre, eye) > 0]
        ax.computed_zorder = False
        collection.set_zorder(1)
        ax.add_collection3d(Line3DCollection(front, colors="#1b1b1b", linewidths=0.8,
                                             zorder=2))
    ax.view_init(elev=view[0], azim=view[1])
    reach = max(np.abs(np.concatenate(shifted)).max(), 1e-9)
    for set_limit in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_limit(-reach, reach)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def figures(directory=RESULTS_DIR / "figures"):
    """Cells of the classical candidates, flat and relaxed, as one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    panels = []
    for name, cell, colour, title in (
        ("fcc", 0, "#c9a227", "FCC: rhombic dodecahedron\n5.3454 (densest packing)"),
        ("hcp", 0, "#d98c3f", "HCP: trapezo-rhombic dodecahedron\n5.3454 (densest packing)"),
        ("bcc", 0, "#4f86c6", "BCC: truncated octahedron\n5.3147 flat"),
    ):
        box, points = point_set(name)
        panels.append((cell_polygons(periodic_voronoi(points, box), cell), colour, title,
                       True, ()))
    box, points = point_set("bcc")
    mesh = triangulate(periodic_voronoi(points, box), 2)
    kelvin = relax(mesh)
    triangles, lines = cell_triangles(mesh, kelvin["y"], 0)
    panels.append((list(triangles), "#4f86c6", "Kelvin, curved (relaxed)\n5.3063", False,
                   lines))
    box, points = point_set("a15")
    mesh = triangulate(periodic_voronoi(points, box), 2)
    phelan = relax(mesh)
    for cell, colour, title in ((0, "#5aa469", "Weaire-Phelan: 12-face cell\n(relaxed)"),
                                (2, "#3f8f6b", "Weaire-Phelan: 14-face cell\nfoam 5.2883")):
        triangles, lines = cell_triangles(mesh, phelan["y"], cell)
        panels.append((list(triangles), colour, title, False, lines))
    fig = plt.figure(figsize=(12, 7.5))
    for k, (polygons, colour, title, edges, lines) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        draw(ax, polygons, colour, title, edges, lines, view=(22, 35))
    fig.suptitle("The Kelvin problem: cost = cell surface / volume^(2/3)  (sphere 4.836)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = directory / "exp042_kelvin_cells.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def axis_choice(foam, axis):
    """A ``choose`` for ``open_six_cell_vertices``: every square perpendicular to
    the same coordinate axis."""
    box, points = foam["box"], foam["points"]
    target = np.eye(3)[axis]

    def choose(vid, pairs):
        scores = []
        for c1, c2 in pairs:
            d = (points[c1[0]] + np.array(c1[1]) @ box) - (points[c2[0]] + np.array(c2[1]) @ box)
            scores.append(abs(np.dot(d / np.linalg.norm(d), target)))
        return int(np.argmax(scores))

    return choose


def densest_packing_decay(level=2):
    """Open the six-cell vertices of the FCC and HCP foams and relax with the
    box free. Returns cost, final box and the box's shape parameters."""
    out = {}
    for name in ("fcc", "hcp"):
        box, points = point_set(name)
        foam = periodic_voronoi(points, box)
        choose = axis_choice(foam, 2) if name == "fcc" else (lambda vid, pairs: 0)
        opened = open_six_cell_vertices(foam, choose)
        result = relax(triangulate(opened, level), free_box=True)
        lengths = np.linalg.norm(result["box"], axis=1)
        out[name] = {"flat_cost_after_opening": flat_cost(opened)[0],
                     "relaxed_cost": result["cost"], "box": result["box"].tolist(),
                     "box_lengths": lengths.tolist(),
                     "volume_error": result["volume_error"]}
    return out


def decay_figure(directory=RESULTS_DIR / "figures"):
    """FCC's rhombic dodecahedron, opened as a soap film opens it, and what it
    relaxes into."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box, points = point_set("fcc")
    foam = periodic_voronoi(points, box)
    opened = open_six_cell_vertices(foam, axis_choice(foam, 2), size=0.08)
    mesh = triangulate(opened, 2)
    result = relax(mesh, free_box=True)
    relaxed_mesh = dict(mesh, box=result["box"])
    triangles, lines = cell_triangles(relaxed_mesh, result["y"], 0)
    fig = plt.figure(figsize=(12, 4.4))
    panels = (
        (cell_polygons(foam, 0), "#c9a227", "FCC: rhombic dodecahedron\n6 cells meet at 6 corners", True, ()),
        (cell_polygons(opened, 0), "#d9b54a", "corners opened into squares\n(Plateau's cube frame)", True, ()),
        (list(triangles), "#4f86c6", f"relaxed, box free: {result['cost']:.4f}\n= Kelvin's foam (Bain path)", False, lines),
    )
    for k, (polygons, colour, title, edges, segs) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw(ax, polygons, colour, title, edges, segs, view=(18, 30))
    fig.suptitle("The densest-packing foam is not a soap foam: it decays into Kelvin's", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = directory / "exp042_fcc_decays_to_kelvin.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


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
    print("\nRelaxed to curved soap films at equal volumes (levels 1, 2, 3; extrapolated):")
    relaxed = {}
    for name in ("bcc", "a15"):
        result = relaxed_cost(name, levels=(1, 2, 3))
        relaxed[name] = result
        print(f"  {name:>4}: {[round(c, 6) for c in result['costs']]} -> "
              f"{result['extrapolated']:.5f}  (volume error {result['volume_error']:.0e})")
    print("\nThe densest-packing foams with their six-cell vertices opened, box free (level 2):")
    decay = densest_packing_decay(2)
    for name, result in decay.items():
        print(f"  {name}: flat {result['flat_cost_after_opening']:.5f} -> relaxed "
              f"{result['relaxed_cost']:.6f}, box edge lengths "
              f"{[round(x, 4) for x in result['box_lengths']]}")
    pictures = [str(figures()), str(decay_figure())]
    print("figures:", *pictures)
    path = RESULTS_DIR / "exp042_kelvin_problem.json"
    path.write_text(json.dumps({"sphere_bound": SPHERE, "exact": rows, "voronoi": foams,
                                "relaxed": relaxed, "densest_packing_decay": decay},
                               indent=1, default=float))
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
