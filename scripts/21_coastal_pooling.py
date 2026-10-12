"""
Agulhas Region — Coastal Cell Pooling

Pools thin neighbouring coastal cells into contiguous coastal segments to
reduce the sampling problem near shore.

A coastal cell is defined in the same way as elsewhere in the pipeline:
an ocean cell with a non-ocean neighbour on the grid.

The pooling is deliberately separate from the main transition operator.
It does not overwrite P. Instead it creates a shared coastal segmentation
for drogued and undrogued drifters that can be used for coastal-risk
summaries and the Week 6 traffic-weighted coastal-segment analysis.

Segments are built along the coastal-cell adjacency graph. Starting from
individual coastal cells, a segment with fewer than MIN_DRIFTERS distinct
drifters in either drogue class is merged with an adjacent coastal segment.
Merging stops when both classes reach MIN_DRIFTERS, when no adjacent segment
is available, or when MAX_POOL_CELLS would be exceeded.

Distinct drifters are reconstructed directly from the GDP transitions rather
than obtained by summing row_drifters, because one drifter can visit several
neighbouring coastal cells.

Baseline run:

    python scripts/21_coastal_pooling.py --res 1 --tau 3.5
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config import (
    DROGUE_TYPES,
    MIN_DRIFTERS,
    grid_shape,
    operator_path,
    parse_args,
    setting_tag,
)

from transport import (
    EXIT_LABELS,
    cell_centres,
    coastal_states,
    load_operator,
    move_indices,
    strand_times,
)


# Maximum number of neighbouring 1° coastal cells allowed in one pooled
# segment. This prevents very sparse coastlines from collapsing into one
# extremely long segment.
MAX_POOL_CELLS = 4


# ── 1. Settings ───────────────────────────────────────────────────────────────
args = parse_args(
    __doc__,
    grid=True,
    lag=True,
)

RES = args.res
TAU = args.tau

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box

TAG = setting_tag(
    RES,
    TAU,
    args.box,
)

CELL_FILE = (
    f"data/coastal_pooling_cells_{TAG}.csv"
)

SEGMENT_FILE = (
    f"data/coastal_pooling_segments_{TAG}.csv"
)

SUMMARY_FILE = (
    f"data/coastal_pooling_summary_{TAG}.csv"
)

MAPPING_FILE = (
    f"data/coastal_segments_{TAG}.npz"
)

FIGURE_FILE = (
    f"figures/agulhas_coastal_pooling_{TAG}.png"
)

os.makedirs(
    "data",
    exist_ok=True,
)

os.makedirs(
    "figures",
    exist_ok=True,
)


print(
    "========================================"
)

print(
    "COASTAL CELL POOLING"
)

print(
    "========================================"
)

print(
    f"Region              : {args.box}"
)

print(
    f"Grid                : {RES:g}°"
)

print(
    f"Lag                 : {TAU:g} days"
)

print(
    f"Reliability target  : {MIN_DRIFTERS} distinct drifters"
)

print(
    f"Maximum segment size: {MAX_POOL_CELLS} coastal cells"
)


# ── 2. Load annual operators ──────────────────────────────────────────────────
ops = {}

for name in DROGUE_TYPES:

    path = operator_path(
        name,
        RES,
        TAU,
        args.box,
    )

    if not os.path.exists(
        path
    ):
        raise SystemExit(
            f"\nMissing {path}\n"
            f"Build the annual matrix first with:\n"
            f"  python scripts/03_transition_matrix.py "
            f"--res {RES:g} --tau {TAU:g}"
        )

    ops[
        name
    ] = load_operator(
        path
    )


reference_name = next(
    iter(
        DROGUE_TYPES
    )
)

reference = ops[
    reference_name
]

cells = reference[
    "cell_flat"
].astype(
    int
)

n_cells = len(
    cells
)

n_lon = (
    len(
        reference[
            "lon_edges"
        ]
    )
    - 1
)

n_lat = (
    len(
        reference[
            "lat_edges"
        ]
    )
    - 1
)


# Both drogue types must use the same ocean-state definition.
for name, op in ops.items():

    if not np.array_equal(
        op[
            "cell_flat"
        ],
        cells,
    ):
        raise SystemExit(
            f"Ocean states differ for {name}. "
            "Rebuild the annual matrices with script 03."
        )

    if not np.array_equal(
        op[
            "exit_labels"
        ].astype(str),
        reference[
            "exit_labels"
        ].astype(str),
    ):
        raise SystemExit(
            f"Absorbing states differ for {name}. "
            "Rebuild the annual matrices with script 03."
        )


coast = coastal_states(
    reference
)

coastal_states_idx = np.where(
    coast
)[0]

print(
    f"Ocean cells         : {n_cells}"
)

print(
    f"Coastal cells       : {len(coastal_states_idx)}"
)


# ── 3. Reconstruct distinct drifters behind every source row ─────────────────
#
# We need the actual drifter IDs. Summing row_drifters from neighbouring rows
# would over-count a drifter that visited more than one coastal cell.
ds = xr.open_dataset(
    args.data
)

lon = ds[
    "lon"
].values.astype(
    float
)

lat = ds[
    "lat"
].values.astype(
    float
)

drogue = ds[
    "drogue_status"
].values.astype(
    bool
)

t = (
    ds[
        "time"
    ]
    .values
    .astype(
        "datetime64[s]"
    )
    .astype(
        np.int64
    )
)

traj_idx = np.repeat(
    np.arange(
        ds.sizes[
            "traj"
        ]
    ),
    ds[
        "rowsize"
    ].values,
)

in_R = (
    (lon >= LON_MIN)
    & (lon < LON_MAX)
    & (lat >= LAT_MIN)
    & (lat < LAT_MAX)
)

strand_t = strand_times(
    ds,
    traj_idx,
    t,
    in_R,
)

start, end, stranded = move_indices(
    traj_idx,
    t,
    in_R,
    TAU,
    strand_t,
)


# Observation -> raw grid cell.
cell_flat_obs = np.where(
    in_R,
    (
        (
            (lat - LAT_MIN)
            // RES
        ).astype(int)
        * n_lon
        +
        (
            (lon - LON_MIN)
            // RES
        ).astype(int)
    ),
    -1,
)


# Raw grid cell -> operator state.
state_of_cell = np.full(
    n_lon * n_lat,
    -1,
    dtype=int,
)

state_of_cell[
    cells
] = np.arange(
    n_cells
)


drifters_by_state = {
    name: [
        set()
        for _ in range(
            n_cells
        )
    ]
    for name in DROGUE_TYPES
}


for name, flag in DROGUE_TYPES.items():

    same_type = (
        (drogue[start] == flag)
        & (drogue[end] == flag)
    )

    s = start[
        same_type
    ]

    from_state = state_of_cell[
        cell_flat_obs[
            s
        ]
    ]

    pairs = np.unique(
        np.column_stack(
            [
                from_state,
                traj_idx[
                    s
                ],
            ]
        ),
        axis=0,
    )

    for st, drifter in pairs:

        drifters_by_state[
            name
        ][
            int(st)
        ].add(
            int(
                drifter
            )
        )


# Check that the reconstruction exactly matches the annual operator.
for name in DROGUE_TYPES:

    reconstructed = np.array(
        [
            len(
                x
            )
            for x in drifters_by_state[
                name
            ]
        ],
        dtype=int,
    )

    saved = ops[
        name
    ][
        "row_drifters"
    ].astype(
        int
    )

    if not np.array_equal(
        reconstructed,
        saved,
    ):
        mismatch = np.sum(
            reconstructed
            != saved
        )

        raise SystemExit(
            f"\nDrifter-count mismatch for {name}: "
            f"{mismatch} rows differ from the saved operator.\n"
            "Rebuild script 03 using the current data file before running 21."
        )


print(
    "Drifter reconstruction: matches saved annual matrices"
)


# ── 4. Coastal adjacency graph ────────────────────────────────────────────────
#
# Eight-neighbour connectivity keeps diagonally connected pieces of coastline
# together while still restricting pooling to immediately neighbouring
# coastal cells.
coastal_by_flat = {
    int(
        cells[
            st
        ]
    ): int(
        st
    )
    for st in coastal_states_idx
}

neighbours = {
    int(
        st
    ): set()
    for st in coastal_states_idx
}


for st in coastal_states_idx:

    flat = int(
        cells[
            st
        ]
    )

    j, i = divmod(
        flat,
        n_lon,
    )

    for dj in (
        -1,
        0,
        1,
    ):

        for di in (
            -1,
            0,
            1,
        ):

            if (
                dj == 0
                and di == 0
            ):
                continue

            jj = j + dj
            ii = i + di

            if (
                jj < 0
                or jj >= n_lat
                or ii < 0
                or ii >= n_lon
            ):
                continue

            other_flat = (
                jj * n_lon
                + ii
            )

            if other_flat in coastal_by_flat:

                neighbours[
                    int(
                        st
                    )
                ].add(
                    coastal_by_flat[
                        other_flat
                    ]
                )


# ── 5. Shared coastal segmentation ───────────────────────────────────────────
#
# Start with one segment per coastal cell. Thin segments are greedily merged
# with an adjacent segment. The chosen neighbour is the one that gives the
# strongest minimum support across the two drogue classes.
segments = {
    int(
        st
    ): {
        int(
            st
        )
    }
    for st in coastal_states_idx
}

segment_of = {
    int(
        st
    ): int(
        st
    )
    for st in coastal_states_idx
}


def pooled_drifters(
    members,
    name,
):
    """Unique drifters contributing moves from any cell in members."""

    result = set()

    for st in members:

        result.update(
            drifters_by_state[
                name
            ][
                st
            ]
        )

    return result


def support_counts(
    members,
):
    """Distinct-drifer support for all drogue classes."""

    return {
        name: len(
            pooled_drifters(
                members,
                name,
            )
        )
        for name in DROGUE_TYPES
    }


def segment_neighbours(
    segment_id,
):
    """Adjacent current segment IDs."""

    result = set()

    for st in segments[
        segment_id
    ]:

        for nb in neighbours[
            st
        ]:

            result.add(
                segment_of[
                    nb
                ]
            )

    result.discard(
        segment_id
    )

    return result


merge_count = 0


while True:

    thin = []

    for segment_id, members in segments.items():

        support = support_counts(
            members
        )

        minimum_support = min(
            support.values()
        )

        if minimum_support < MIN_DRIFTERS:

            thin.append(
                (
                    minimum_support,
                    len(
                        members
                    ),
                    segment_id,
                )
            )

    thin.sort()

    merged_this_round = False

    for _, _, segment_id in thin:

        if segment_id not in segments:
            continue

        current_members = segments[
            segment_id
        ]

        candidates = []

        for other_id in segment_neighbours(
            segment_id
        ):

            if other_id not in segments:
                continue

            combined = (
                current_members
                | segments[
                    other_id
                ]
            )

            if (
                len(
                    combined
                )
                > MAX_POOL_CELLS
            ):
                continue

            support = support_counts(
                combined
            )

            minimum_support = min(
                support.values()
            )

            total_support = sum(
                support.values()
            )

            # Prefer the merge that most improves the weakest drogue class.
            # Ties prefer more total support, then fewer cells.
            score = (
                minimum_support,
                total_support,
                -len(
                    combined
                ),
                -min(
                    combined
                ),
            )

            candidates.append(
                (
                    score,
                    other_id,
                    combined,
                )
            )

        if not candidates:
            continue

        candidates.sort(
            reverse=True,
            key=lambda x: x[0],
        )

        _, other_id, combined = candidates[
            0
        ]

        keep_id = min(
            segment_id,
            other_id,
        )

        drop_id = max(
            segment_id,
            other_id,
        )

        segments[
            keep_id
        ] = combined

        if drop_id in segments:
            del segments[
                drop_id
            ]

        if (
            segment_id != keep_id
            and segment_id in segments
        ):
            del segments[
                segment_id
            ]

        for st in combined:

            segment_of[
                st
            ] = keep_id

        merge_count += 1
        merged_this_round = True

        break

    if not merged_this_round:
        break


# Stable segment numbering: order by the smallest raw grid cell.
ordered_segments = sorted(
    segments.values(),
    key=lambda members: min(
        cells[
            list(
                members
            )
        ]
    ),
)

segment_of_state = np.full(
    n_cells,
    -1,
    dtype=int,
)

for new_id, members in enumerate(
    ordered_segments
):

    for st in members:

        segment_of_state[
            st
        ] = new_id


n_segments = len(
    ordered_segments
)

print(
    f"Pooling merges       : {merge_count}"
)

print(
    f"Coastal segments     : {n_segments}"
)


# ── 6. Segment-level support and one-step fates ───────────────────────────────
segment_rows = []

segment_support = {
    name: np.zeros(
        n_segments,
        dtype=int,
    )
    for name in DROGUE_TYPES
}


for segment_id, members in enumerate(
    ordered_segments
):

    members = sorted(
        members
    )

    member_flats = cells[
        members
    ]

    member_j, member_i = np.divmod(
        member_flats,
        n_lon,
    )

    lon_centres = (
        reference[
            "lon_edges"
        ][
            member_i
        ]
        +
        reference[
            "lon_edges"
        ][
            member_i + 1
        ]
    ) / 2

    lat_centres = (
        reference[
            "lat_edges"
        ][
            member_j
        ]
        +
        reference[
            "lat_edges"
        ][
            member_j + 1
        ]
    ) / 2

    for name in DROGUE_TYPES:

        op = ops[
            name
        ]

        unique_drifters = pooled_drifters(
            members,
            name,
        )

        n_drifters = len(
            unique_drifters
        )

        segment_support[
            name
        ][
            segment_id
        ] = n_drifters

        pooled_counts = op[
            "C"
        ][
            members
        ].sum(
            axis=0
        )

        total_moves = pooled_counts.sum()

        if total_moves > 0:

            prob_in_R = (
                pooled_counts[
                    :n_cells
                ].sum()
                / total_moves
            )

            absorbing = (
                pooled_counts[
                    n_cells:
                ]
                / total_moves
            )

        else:

            prob_in_R = np.nan

            absorbing = np.full(
                len(
                    EXIT_LABELS
                ),
                np.nan,
            )

        row = {
            "segment_id":
                segment_id + 1,

            "drogue":
                name,

            "n_cells":
                len(
                    members
                ),

            "mean_lon":
                float(
                    np.mean(
                        lon_centres
                    )
                ),

            "mean_lat":
                float(
                    np.mean(
                        lat_centres
                    )
                ),

            "distinct_drifters":
                n_drifters,

            "reliable":
                bool(
                    n_drifters
                    >= MIN_DRIFTERS
                ),

            "moves":
                int(
                    total_moves
                ),

            "p_in_R":
                float(
                    prob_in_R
                ),
        }

        for k, label in enumerate(
            op[
                "exit_labels"
            ].astype(str)
        ):

            row[
                f"p_{label}"
            ] = float(
                absorbing[
                    k
                ]
            )

        segment_rows.append(
            row
        )


segment_table = pd.DataFrame(
    segment_rows
)

segment_table.to_csv(
    SEGMENT_FILE,
    index=False,
)


# ── 7. Cell -> segment mapping ────────────────────────────────────────────────
cell_lon, cell_lat = cell_centres(
    reference
)

cell_rows = []


for st in coastal_states_idx:

    segment_id = segment_of_state[
        st
    ]

    members = ordered_segments[
        segment_id
    ]

    row = {
        "state":
            int(
                st
            ),

        "cell_flat":
            int(
                cells[
                    st
                ]
            ),

        "lon":
            float(
                cell_lon[
                    st
                ]
            ),

        "lat":
            float(
                cell_lat[
                    st
                ]
            ),

        "segment_id":
            int(
                segment_id
                + 1
            ),

        "segment_cells":
            int(
                len(
                    members
                )
            ),
    }

    for name in DROGUE_TYPES:

        original_count = len(
            drifters_by_state[
                name
            ][
                st
            ]
        )

        pooled_count = segment_support[
            name
        ][
            segment_id
        ]

        row[
            f"{name}_cell_drifters"
        ] = int(
            original_count
        )

        row[
            f"{name}_cell_reliable"
        ] = bool(
            original_count
            >= MIN_DRIFTERS
        )

        row[
            f"{name}_pooled_drifters"
        ] = int(
            pooled_count
        )

        row[
            f"{name}_pooled_reliable"
        ] = bool(
            pooled_count
            >= MIN_DRIFTERS
        )

    row[
        "pooled_reliable_both"
    ] = bool(
        all(
            segment_support[
                name
            ][
                segment_id
            ]
            >= MIN_DRIFTERS
            for name in DROGUE_TYPES
        )
    )

    cell_rows.append(
        row
    )


cell_table = pd.DataFrame(
    cell_rows
)

cell_table.to_csv(
    CELL_FILE,
    index=False,
)


# ── 8. Summary before and after pooling ───────────────────────────────────────
summary_rows = []


for name in DROGUE_TYPES:

    original = np.array(
        [
            len(
                drifters_by_state[
                    name
                ][
                    st
                ]
            )
            for st in coastal_states_idx
        ],
        dtype=int,
    )

    original_active = (
        original
        > 0
    )

    if original_active.any():

        original_reliable_share = float(
            np.mean(
                original[
                    original_active
                ]
                >= MIN_DRIFTERS
            )
        )

    else:

        original_reliable_share = np.nan

    pooled_by_cell = np.array(
        [
            segment_support[
                name
            ][
                segment_of_state[
                    st
                ]
            ]
            for st in coastal_states_idx
        ],
        dtype=int,
    )

    pooled_cell_reliable_share = float(
        np.mean(
            pooled_by_cell
            >= MIN_DRIFTERS
        )
    )

    pooled_segment_reliable_share = float(
        np.mean(
            segment_support[
                name
            ]
            >= MIN_DRIFTERS
        )
    )

    summary_rows.append(
        {
            "drogue":
                name,

            "res":
                RES,

            "tau_days":
                TAU,

            "coastal_cells":
                len(
                    coastal_states_idx
                ),

            "segments":
                n_segments,

            "max_pool_cells":
                MAX_POOL_CELLS,

            "original_active_cells":
                int(
                    original_active.sum()
                ),

            "original_reliable_active_share":
                original_reliable_share,

            "pooled_reliable_cell_share":
                pooled_cell_reliable_share,

            "pooled_reliable_segment_share":
                pooled_segment_reliable_share,

            "median_segment_support":
                float(
                    np.median(
                        segment_support[
                            name
                        ]
                    )
                ),

            "min_segment_support":
                int(
                    np.min(
                        segment_support[
                            name
                        ]
                    )
                ),

            "max_segment_support":
                int(
                    np.max(
                        segment_support[
                            name
                        ]
                    )
                ),
        }
    )


summary_table = pd.DataFrame(
    summary_rows
)

summary_table.to_csv(
    SUMMARY_FILE,
    index=False,
)


reliable_both_segments = np.ones(
    n_segments,
    dtype=bool,
)

for name in DROGUE_TYPES:

    reliable_both_segments &= (
        segment_support[
            name
        ]
        >= MIN_DRIFTERS
    )


segment_sizes = np.array(
    [
        len(
            x
        )
        for x in ordered_segments
    ]
)


print(
    "\n----------------------------------------"
)

print(
    "SUPPORT BEFORE / AFTER POOLING"
)

print(
    "----------------------------------------"
)


for row in summary_table.itertuples():

    print(
        f"\n{row.drogue.upper()}"
    )

    print(
        f"Original reliable coastal rows "
        f"(active only) : "
        f"{row.original_reliable_active_share:.1%}"
    )

    print(
        f"Coastal cells covered by a "
        f"reliable pooled segment: "
        f"{row.pooled_reliable_cell_share:.1%}"
    )

    print(
        f"Reliable pooled segments       : "
        f"{row.pooled_reliable_segment_share:.1%}"
    )

    print(
        f"Median pooled support          : "
        f"{row.median_segment_support:.0f} drifters"
    )


print(
    "\nSHARED SEGMENTS"
)

print(
    f"Reliable in both types : "
    f"{reliable_both_segments.sum()} / "
    f"{n_segments} "
    f"({reliable_both_segments.mean():.1%})"
)

print(
    f"Median segment size    : "
    f"{np.median(segment_sizes):.1f} cells"
)

print(
    f"Maximum segment size   : "
    f"{segment_sizes.max()} cells"
)

print(
    f"Segments still thin    : "
    f"{(~reliable_both_segments).sum()}"
)


# ── 9. Save mapping for Week 6 ────────────────────────────────────────────────
np.savez_compressed(
    MAPPING_FILE,
    segment_of_state=segment_of_state,
    coastal=coast,
    cell_flat=cells,
    drogued_support=segment_support[
        "drogued"
    ],
    undrogued_support=segment_support[
        "undrogued"
    ],
    segment_size=segment_sizes,
    reliable_both=reliable_both_segments,
    res=RES,
    tau_days=TAU,
    box=np.array(
        args.box
    ),
    min_drifters=MIN_DRIFTERS,
    max_pool_cells=MAX_POOL_CELLS,
)


# ── 10. Figure ────────────────────────────────────────────────────────────────
segment_grid = np.full(
    n_lon * n_lat,
    np.nan,
)

drogued_grid = np.full(
    n_lon * n_lat,
    np.nan,
)

undrogued_grid = np.full(
    n_lon * n_lat,
    np.nan,
)


for st in coastal_states_idx:

    flat = cells[
        st
    ]

    segment_id = segment_of_state[
        st
    ]

    segment_grid[
        flat
    ] = (
        segment_id
        + 1
    )

    drogued_grid[
        flat
    ] = segment_support[
        "drogued"
    ][
        segment_id
    ]

    undrogued_grid[
        flat
    ] = segment_support[
        "undrogued"
    ][
        segment_id
    ]


segment_grid = segment_grid.reshape(
    n_lat,
    n_lon,
)

drogued_grid = drogued_grid.reshape(
    n_lat,
    n_lon,
)

undrogued_grid = undrogued_grid.reshape(
    n_lat,
    n_lon,
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        16,
        5,
    ),
    sharex=True,
    sharey=True,
)


extent = [
    reference[
        "lon_edges"
    ][0],
    reference[
        "lon_edges"
    ][-1],
    reference[
        "lat_edges"
    ][0],
    reference[
        "lat_edges"
    ][-1],
]


im0 = axes[
    0
].imshow(
    segment_grid,
    origin="lower",
    extent=extent,
    aspect="auto",
)

axes[
    0
].set_title(
    "Pooled coastal segment ID"
)

fig.colorbar(
    im0,
    ax=axes[
        0
    ],
    shrink=0.8,
)


im1 = axes[
    1
].imshow(
    drogued_grid,
    origin="lower",
    extent=extent,
    aspect="auto",
    vmin=0,
)

axes[
    1
].set_title(
    "Drogued: distinct drifters\nper pooled segment"
)

fig.colorbar(
    im1,
    ax=axes[
        1
    ],
    shrink=0.8,
)


im2 = axes[
    2
].imshow(
    undrogued_grid,
    origin="lower",
    extent=extent,
    aspect="auto",
    vmin=0,
)

axes[
    2
].set_title(
    "Undrogued: distinct drifters\nper pooled segment"
)

fig.colorbar(
    im2,
    ax=axes[
        2
    ],
    shrink=0.8,
)


for ax in axes:

    ax.set_xlabel(
        "Longitude"
    )

axes[
    0
].set_ylabel(
    "Latitude"
)


fig.suptitle(
    f"Coastal pooling — "
    f"{RES:g}°, "
    f"τ = {TAU:g} d"
)

plt.savefig(
    FIGURE_FILE,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ── 11. Outputs ───────────────────────────────────────────────────────────────
print(
    "\nSaved:"
)

print(
    f"  {CELL_FILE}"
)

print(
    f"  {SEGMENT_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)

print(
    f"  {MAPPING_FILE}"
)

print(
    f"  {FIGURE_FILE}"
)

print(
    "\nInterpretation:"
)

print(
    "Pooling is used for coastal-segment summaries and future coastal-risk "
    "analysis. It does not replace or smooth the main transition matrix."
)

print(
    "Segments that remain below the drifter threshold are retained and "
    "explicitly flagged rather than being merged over an arbitrarily long "
    "stretch of coastline."
)
