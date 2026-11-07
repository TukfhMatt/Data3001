"""
Agulhas Region — Coastal Cell Pooling

Pools thin neighbouring coastal cells into contiguous coastal segments to
reduce the sampling problem near shore.

A coastal candidate is an ocean cell next to a non-model-ocean cell on
the grid. For this coastal-pooling analysis ONLY, such candidates are
conservatively filtered by Natural Earth 50m land polygons: mapped land
must intersect the candidate's cell or one of its eight neighbours.
This removes suspect offshore candidates; it does not exhaustively
identify all coastal ocean cells. shipping_overlay, bootstrap, oil_wind and
coastal_coverage use the model-neighbour coastal definition.

The pooling is deliberately separate from the main transition operator.
It does not overwrite P. Instead it creates a shared coastal segmentation
for drogued and undrogued drifters that can be used for coastal-risk
summaries and the traffic-weighted coastal-segment analysis in traffic_exposure.

Segments are built along the coastal-cell adjacency graph. Starting from
individual coastal cells, a segment with fewer than MIN_DRIFTERS distinct
drifters in either drogue class is merged with an adjacent coastal segment.
Merging stops when both classes reach MIN_DRIFTERS, when no adjacent segment
is available, or when MAX_POOL_CELLS would be exceeded.

Distinct drifters are reconstructed directly from the GDP transitions rather
than obtained by summing row_drifters, because one drifter can visit several
neighbouring coastal cells.

Baseline run:

    python -m agulhas coastal_pooling --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, MIN_DRIFTERS, operator_path, parse_args, region_prefix, setting_tag
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import save_figure
from agulhas.transport import EXIT_LABELS, cell_centres, coastal_states, load_operator

# Maximum number of neighbouring 1° coastal cells allowed in one pooled
# segment. This prevents very sparse coastlines from collapsing into one
# extremely long segment.
MAX_POOL_CELLS = 4
COAST_DEFINITION = "original_model_neighbours_and_NaturalEarth50m_land_8neighbours"


def print_header(args):
    print("========================================")
    print("COASTAL CELL POOLING")
    print("========================================")
    print(f"Region              : {args.box}")
    print(f"Grid                : {args.res:g}°")
    print(f"Lag                 : {args.tau:g} days")
    print(f"Reliability target  : {MIN_DRIFTERS} distinct drifters")
    print(f"Maximum segment size: {MAX_POOL_CELLS} coastal cells")


def load_ops(args):
    """Both drogue types' matrices, which must share their ocean and absorbing states."""
    ops = {}
    for name in DROGUE_TYPES:
        path = operator_path(name, args.res, args.tau, args.box)
        if not os.path.exists(path):
            raise SystemExit(f"\nMissing {path}\nBuild the annual matrix first with:\n"
                             f"  python -m agulhas transition_matrix --res {args.res:g} --tau {args.tau:g}")
        ops[name] = load_operator(path)
    reference = ops[next(iter(DROGUE_TYPES))]
    for name, op in ops.items():
        if not np.array_equal(op["cell_flat"], reference["cell_flat"]):
            raise SystemExit(f"Ocean states differ for {name}. Rebuild the annual matrices with transition_matrix.")
        if not np.array_equal(op["exit_labels"].astype(str), reference["exit_labels"].astype(str)):
            raise SystemExit(f"Absorbing states differ for {name}. Rebuild the annual matrices with transition_matrix.")
    return ops


# ── Coastal cells: model neighbours near Natural Earth land ─────────────────

def natural_earth_land(box, res):
    """Natural Earth 50m land around the box, prepared for fast intersection tests."""
    try:
        import cartopy.io.shapereader as ne_reader
        from shapely.geometry import box as geo_box
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError as exc:
        raise SystemExit("Geographic review needs cartopy and shapely. Install with:\n"
                         "  python -m pip install cartopy shapely") from exc
    lon_min, lon_max, lat_min, lat_max = box
    try:
        ne_bounds = geo_box(lon_min - 2 * res, lat_min - 2 * res, lon_max + 2 * res, lat_max + 2 * res)
        ne_path = ne_reader.natural_earth(resolution="50m", category="physical", name="land")
        land_pieces = [geom for geom in ne_reader.Reader(ne_path, bbox=ne_bounds.bounds).geometries()
                       if not geom.is_empty and geom.intersects(ne_bounds)]
        if not land_pieces:
            raise ValueError("No Natural Earth land polygons overlap the study area")
        return prep(unary_union(land_pieces)), geo_box
    except Exception as exc:
        raise SystemExit(f"Failed to load Natural Earth 50m land data: {exc}\n"
                         "The first run may need an internet connection.") from exc


def coastal_cells(reference, args):
    """The model-neighbour coastal candidates, kept only if Natural Earth land is in
    the same or an immediately adjacent grid cell. This removes obvious offshore
    artefacts caused by confusing unobserved sea cells with land; it is not a full
    independent redefinition of all physical coastal cells."""
    cells = reference["cell_flat"].astype(int)
    lon_edges, lat_edges = reference["lon_edges"].astype(float), reference["lat_edges"].astype(float)
    n_lon, n_lat = len(lon_edges) - 1, len(lat_edges) - 1
    candidates = coastal_states(reference)
    land, geo_box = natural_earth_land(args.box, args.res)
    land_in_cell = np.zeros((n_lat, n_lon), dtype=bool)
    for jj in range(n_lat):
        for ii in range(n_lon):
            land_in_cell[jj, ii] = land.intersects(geo_box(lon_edges[ii], lat_edges[jj], lon_edges[ii + 1], lat_edges[jj + 1]))
    pad = np.pad(land_in_cell, 1, constant_values=False)
    near_land = np.zeros_like(land_in_cell)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            near_land |= pad[1 + dj: 1 + dj + n_lat, 1 + di: 1 + di + n_lon]

    near_states = near_land.ravel()[cells]
    coast = candidates & near_states
    rejected = np.flatnonzero(candidates & ~near_states)
    print(f"Natural Earth 50m coastal filter: {candidates.sum()} original candidates; "
          f"{coast.sum()} retained; {len(rejected)} removed as not near mapped land")
    for state in rejected:
        jj, ii = divmod(int(cells[state]), n_lon)
        print(f"  excluded state {state}: lon={(lon_edges[ii] + lon_edges[ii + 1]) / 2:.3f}, "
              f"lat={(lat_edges[jj] + lat_edges[jj + 1]) / 2:.3f}")
    print("WARNING: This is a conservative FILTER of the existing candidate set; "
          "near-land cells outside that set are not added.")
    return coast


# ── Distinct drifters behind every source row ────────────────────────────────

def drifters_by_state(moves, ops):
    """Per drogue type and ocean state, the set of drifters with moves from it.
    Summing row_drifters from neighbouring rows would over-count a drifter that
    visited more than one coastal cell. Stops unless the sets rebuild row_drifters."""
    cells = ops["drogued"]["cell_flat"].astype(int)
    state_of_cell = np.full(moves.grid.n_cells, -1, dtype=int)
    state_of_cell[cells] = np.arange(len(cells))
    by_state = {name: [set() for _ in range(len(cells))] for name in DROGUE_TYPES}
    for name, flag in DROGUE_TYPES.items():
        s = moves.of_type(flag)[0]
        pairs = np.unique(np.column_stack([state_of_cell[moves.cell_flat[s]], moves.drifters.traj_idx[s]]), axis=0)
        for st, drifter in pairs:
            by_state[name][int(st)].add(int(drifter))
    for name in DROGUE_TYPES:
        reconstructed = np.array([len(x) for x in by_state[name]], dtype=int)
        saved = ops[name]["row_drifters"].astype(int)
        if not np.array_equal(reconstructed, saved):
            raise SystemExit(f"\nDrifter-count mismatch for {name}: {np.sum(reconstructed != saved)} rows differ "
                             "from the saved operator.\nRebuild it with transition_matrix from the current data file.")
    print("Drifter reconstruction: matches saved annual matrices")
    return by_state


def coastal_neighbours(cells, coastal_idx, n_lon, n_lat):
    """Eight-neighbour adjacency of the coastal states, which keeps diagonally connected
    pieces of coastline together while pooling only immediate neighbours."""
    coastal_by_flat = {int(cells[st]): int(st) for st in coastal_idx}
    neighbours = {int(st): set() for st in coastal_idx}
    for st in coastal_idx:
        j, i = divmod(int(cells[st]), n_lon)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                jj, ii = j + dj, i + di
                if (dj == 0 and di == 0) or jj < 0 or jj >= n_lat or ii < 0 or ii >= n_lon:
                    continue
                other_flat = jj * n_lon + ii
                if other_flat in coastal_by_flat:
                    neighbours[int(st)].add(coastal_by_flat[other_flat])
    return neighbours


# ── Shared coastal segmentation ──────────────────────────────────────────────

class SegmentPool:
    """Starts with one segment per coastal cell; thin segments are greedily merged
    with an adjacent segment, choosing the one that gives the strongest minimum
    support across the two drogue classes."""

    def __init__(self, coastal_idx, neighbours, by_state):
        self.neighbours, self.by_state = neighbours, by_state
        self.segments = {int(st): {int(st)} for st in coastal_idx}
        self.segment_of = {int(st): int(st) for st in coastal_idx}
        self.merges = 0

    def pooled_drifters(self, members, name):
        """Unique drifters contributing moves from any cell in members."""
        result = set()
        for st in members:
            result.update(self.by_state[name][st])
        return result

    def support_counts(self, members):
        """Distinct-drifter support for all drogue classes."""
        return {name: len(self.pooled_drifters(members, name)) for name in DROGUE_TYPES}

    def segment_neighbours(self, segment_id):
        """Adjacent current segment IDs."""
        result = set()
        for st in self.segments[segment_id]:
            for nb in self.neighbours[st]:
                result.add(self.segment_of[nb])
        result.discard(segment_id)
        return result

    def best_merge(self, segment_id):
        """The neighbour to merge with: the merge that most improves the weakest drogue
        class, ties going to more total support, then fewer cells. None if none fits."""
        candidates = []
        for other_id in self.segment_neighbours(segment_id):
            if other_id not in self.segments:
                continue
            combined = self.segments[segment_id] | self.segments[other_id]
            if len(combined) > MAX_POOL_CELLS:
                continue
            support = self.support_counts(combined)
            score = (min(support.values()), sum(support.values()), -len(combined), -min(combined))
            candidates.append((score, other_id, combined))
        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1:]

    def merge_once(self):
        """Merge the thinnest segment that can be merged; False if none can."""
        thin = []
        for segment_id, members in self.segments.items():
            minimum_support = min(self.support_counts(members).values())
            if minimum_support < MIN_DRIFTERS:
                thin.append((minimum_support, len(members), segment_id))
        thin.sort()
        for _, _, segment_id in thin:
            if segment_id not in self.segments:
                continue
            best = self.best_merge(segment_id)
            if best is None:
                continue
            other_id, combined = best
            keep_id, drop_id = min(segment_id, other_id), max(segment_id, other_id)
            self.segments[keep_id] = combined
            if drop_id in self.segments:
                del self.segments[drop_id]
            if segment_id != keep_id and segment_id in self.segments:
                del self.segments[segment_id]
            for st in combined:
                self.segment_of[st] = keep_id
            self.merges += 1
            return True
        return False

    def run(self, cells):
        """Merge until no thin segment can be merged. Returns the segments numbered by
        their smallest grid cell, and each ocean state's segment (-1 off the coast)."""
        while self.merge_once():
            pass
        ordered = sorted(self.segments.values(), key=lambda members: min(cells[list(members)]))
        segment_of_state = np.full(len(cells), -1, dtype=int)
        for new_id, members in enumerate(ordered):
            for st in members:
                segment_of_state[st] = new_id
        return ordered, segment_of_state


# ── Tables ───────────────────────────────────────────────────────────────────

def segment_table(ordered, ops, pool):
    """Support and one-step fates of every segment, and each type's support per segment."""
    reference = ops["drogued"]
    cells = reference["cell_flat"].astype(int)
    n_cells, n_lon = len(cells), len(reference["lon_edges"]) - 1
    rows = []
    support = {name: np.zeros(len(ordered), dtype=int) for name in DROGUE_TYPES}
    for segment_id, members in enumerate(ordered):
        members = sorted(members)
        member_j, member_i = np.divmod(cells[members], n_lon)
        lon_centres = (reference["lon_edges"][member_i] + reference["lon_edges"][member_i + 1]) / 2
        lat_centres = (reference["lat_edges"][member_j] + reference["lat_edges"][member_j + 1]) / 2
        for name in DROGUE_TYPES:
            op = ops[name]
            n_drifters = len(pool.pooled_drifters(members, name))
            support[name][segment_id] = n_drifters
            pooled_counts = op["C"][members].sum(axis=0)
            total_moves = pooled_counts.sum()
            if total_moves > 0:
                prob_in_R = pooled_counts[:n_cells].sum() / total_moves
                absorbing = pooled_counts[n_cells:] / total_moves
            else:
                prob_in_R = np.nan
                absorbing = np.full(len(EXIT_LABELS), np.nan)
            row = {"segment_id": segment_id + 1, "drogue": name, "n_cells": len(members),
                   "mean_lon": float(np.mean(lon_centres)), "mean_lat": float(np.mean(lat_centres)),
                   "distinct_drifters": n_drifters, "reliable": bool(n_drifters >= MIN_DRIFTERS),
                   "moves": int(total_moves), "p_in_R": float(prob_in_R)}
            for k, label in enumerate(op["exit_labels"].astype(str)):
                row[f"p_{label}"] = float(absorbing[k])
            rows.append(row)
    return pd.DataFrame(rows), support


def cell_table(coastal_idx, ordered, segment_of_state, support, reference, by_state):
    """Each coastal cell's segment, and its support before and after pooling."""
    cells = reference["cell_flat"].astype(int)
    cell_lon, cell_lat = cell_centres(reference)
    rows = []
    for st in coastal_idx:
        segment_id = segment_of_state[st]
        row = {"state": int(st), "cell_flat": int(cells[st]), "lon": float(cell_lon[st]), "lat": float(cell_lat[st]),
               "segment_id": int(segment_id + 1), "segment_cells": int(len(ordered[segment_id]))}
        for name in DROGUE_TYPES:
            original_count = len(by_state[name][st])
            pooled_count = support[name][segment_id]
            row[f"{name}_cell_drifters"] = int(original_count)
            row[f"{name}_cell_reliable"] = bool(original_count >= MIN_DRIFTERS)
            row[f"{name}_pooled_drifters"] = int(pooled_count)
            row[f"{name}_pooled_reliable"] = bool(pooled_count >= MIN_DRIFTERS)
        row["pooled_reliable_both"] = bool(all(support[name][segment_id] >= MIN_DRIFTERS for name in DROGUE_TYPES))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(coastal_idx, segment_of_state, support, by_state, args):
    """Support before and after pooling, per drogue type."""
    n_segments = len(next(iter(support.values())))
    rows = []
    for name in DROGUE_TYPES:
        original = np.array([len(by_state[name][st]) for st in coastal_idx], dtype=int)
        original_active = original > 0
        original_reliable_share = (float(np.mean(original[original_active] >= MIN_DRIFTERS))
                                   if original_active.any() else np.nan)
        pooled_by_cell = np.array([support[name][segment_of_state[st]] for st in coastal_idx], dtype=int)
        rows.append({
            "drogue": name, "res": args.res, "tau_days": args.tau,
            "coastal_cells": len(coastal_idx), "segments": n_segments, "max_pool_cells": MAX_POOL_CELLS,
            "original_active_cells": int(original_active.sum()),
            "original_reliable_active_share": original_reliable_share,
            "pooled_reliable_cell_share": float(np.mean(pooled_by_cell >= MIN_DRIFTERS)),
            "pooled_reliable_segment_share": float(np.mean(support[name] >= MIN_DRIFTERS)),
            "median_segment_support": float(np.median(support[name])),
            "min_segment_support": int(np.min(support[name])),
            "max_segment_support": int(np.max(support[name])),
        })
    return pd.DataFrame(rows)


def print_summary(summary, reliable_both, segment_sizes):
    print("\n----------------------------------------")
    print("SUPPORT BEFORE / AFTER POOLING")
    print("----------------------------------------")
    for row in summary.itertuples():
        print(f"\n{row.drogue.upper()}")
        print(f"Original reliable coastal rows (active only) : {row.original_reliable_active_share:.1%}")
        print(f"Coastal cells covered by a reliable pooled segment: {row.pooled_reliable_cell_share:.1%}")
        print(f"Reliable pooled segments       : {row.pooled_reliable_segment_share:.1%}")
        print(f"Median pooled support          : {row.median_segment_support:.0f} drifters")
    print("\nSHARED SEGMENTS")
    print(f"Reliable in both types : {reliable_both.sum()} / {len(reliable_both)} ({reliable_both.mean():.1%})")
    print(f"Median segment size    : {np.median(segment_sizes):.1f} cells")
    print(f"Maximum segment size   : {segment_sizes.max()} cells")
    print(f"Segments still thin    : {(~reliable_both).sum()}")


def save_mapping(path, segment_of_state, coast, support, segment_sizes, reliable_both, cells, args):
    """The segmentation, for traffic_exposure."""
    np.savez_compressed(
        path, segment_of_state=segment_of_state, coastal=coast, cell_flat=cells,
        drogued_support=support["drogued"], undrogued_support=support["undrogued"],
        segment_size=segment_sizes, reliable_both=reliable_both,
        res=args.res, tau_days=args.tau, box=np.array(args.box),
        min_drifters=MIN_DRIFTERS, max_pool_cells=MAX_POOL_CELLS, coast_definition=COAST_DEFINITION)


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_pooling(coastal_idx, segment_of_state, support, reference, args, path):
    cells = reference["cell_flat"].astype(int)
    n_lon, n_lat = len(reference["lon_edges"]) - 1, len(reference["lat_edges"]) - 1
    grids = {key: np.full(n_lon * n_lat, np.nan) for key in ("segment", "drogued", "undrogued")}
    for st in coastal_idx:
        segment_id = segment_of_state[st]
        grids["segment"][cells[st]] = segment_id + 1
        grids["drogued"][cells[st]] = support["drogued"][segment_id]
        grids["undrogued"][cells[st]] = support["undrogued"][segment_id]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    extent = [reference["lon_edges"][0], reference["lon_edges"][-1], reference["lat_edges"][0], reference["lat_edges"][-1]]
    for ax, key, title, kw in [(axes[0], "segment", "Pooled coastal segment ID", {}),
                               (axes[1], "drogued", "Drogued: distinct drifters\nper pooled segment", {"vmin": 0}),
                               (axes[2], "undrogued", "Undrogued: distinct drifters\nper pooled segment", {"vmin": 0})]:
        image = ax.imshow(grids[key].reshape(n_lat, n_lon), origin="lower", extent=extent, aspect="auto", **kw)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    for ax in axes:
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.suptitle(f"Coastal pooling — {args.res:g}°, τ = {args.tau:g} d")
    save_figure(path)


def print_outputs(paths):
    print("\nSaved:")
    for path in paths:
        print(f"  {path}")
    print("\nInterpretation:")
    print("Pooling is used for coastal-segment summaries and future coastal-risk "
          "analysis. It does not replace or smooth the main transition matrix.")
    print("Segments that remain below the drifter threshold are retained and "
          "explicitly flagged rather than being merged over an arbitrarily long "
          "stretch of coastline.")


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_cells = f"data/coastal_pooling_cells_{tag}.csv"
    out_segments = f"data/coastal_pooling_segments_{tag}.csv"
    out_summary = f"data/coastal_pooling_summary_{tag}.csv"
    out_mapping = f"data/coastal_segments_{tag}.npz"
    out_png = f"figures/{region_prefix(args.data)}_coastal_pooling_{tag}.png"

    print_header(args)
    ops = load_ops(args)
    reference = ops["drogued"]
    cells = reference["cell_flat"].astype(int)
    coast = coastal_cells(reference, args)
    coastal_idx = np.where(coast)[0]
    print(f"Ocean cells         : {len(cells)}")
    print(f"Coastal cells       : {len(coastal_idx)}")

    moves = MoveSet(Drifters(args.data), args.box, args.res, args.tau)
    by_state = drifters_by_state(moves, ops)
    pool = SegmentPool(coastal_idx, coastal_neighbours(cells, coastal_idx, moves.grid.n_lon, moves.grid.n_lat), by_state)
    ordered, segment_of_state = pool.run(cells)
    print(f"Pooling merges       : {pool.merges}")
    print(f"Coastal segments     : {len(ordered)}")

    segments, support = segment_table(ordered, ops, pool)
    segments.to_csv(out_segments, index=False)
    cell_table(coastal_idx, ordered, segment_of_state, support, reference, by_state).to_csv(out_cells, index=False)
    summary = summary_table(coastal_idx, segment_of_state, support, by_state, args)
    summary.to_csv(out_summary, index=False)
    reliable_both = np.logical_and.reduce([support[name] >= MIN_DRIFTERS for name in DROGUE_TYPES])
    segment_sizes = np.array([len(x) for x in ordered])
    print_summary(summary, reliable_both, segment_sizes)
    save_mapping(out_mapping, segment_of_state, coast, support, segment_sizes, reliable_both, cells, args)

    plot_pooling(coastal_idx, segment_of_state, support, reference, args, out_png)
    print_outputs([out_cells, out_segments, out_summary, out_mapping, out_png])


if __name__ == "__main__":
    main()
