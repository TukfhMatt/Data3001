"""
One-step moves of a subset on one grid and lag: what every step that rebuilds or
resamples a matrix starts from.

    moves = MoveSet(drifters, args.box, res, tau)
    counts = moves.counts_for(op, name, flag)        # checked against the saved matrix
    P, _ = row_normalise(resampled_counts(counts, n_states, rng), op["cell_flat"], moves.grid.n_lon)
"""

import numpy as np

from agulhas.drifters import in_box
from agulhas.grid import Grid
from agulhas.transport import drifter_counts, same_drogue, transition_states


class MoveSet:
    """All moves starting inside the box, with the grid cell of every observation.
    Give strand_t when the step has it already, to save working it out again."""

    def __init__(self, drifters, box, res, tau, strand_t=None):
        self.drifters = drifters
        self.box = box
        self.tau = tau
        self.grid = Grid(box, res)
        self.in_R = in_box(drifters.lon, drifters.lat, box)
        self.cell_flat = self.grid.cell_of(drifters.lon, drifters.lat, self.in_R)
        self.start, self.end, self.stranded = drifters.moves(self.in_R, tau, strand_t)

    def of_type_mask(self, flag):
        """Boolean per move: it counts for one drogue type (transport.same_drogue)."""
        return same_drogue(self.drifters.drogue, self.start, self.end, self.stranded, flag)

    def of_type(self, flag):
        """Moves that count for one drogue type: (start, end, stranded)."""
        same = self.of_type_mask(flag)
        return self.start[same], self.end[same], self.stranded[same]

    def states(self, cells, start, end, stranded):
        """(from_state, to_state) of the given moves, for a matrix with these ocean cells."""
        d = self.drifters
        return transition_states(self.cell_flat, cells, start, end, d.lon, d.lat, self.box, stranded)

    def counts_for(self, op, name, flag):
        """A saved matrix's moves per drifter, as transport.drifter_counts groups them.
        Stops if they do not rebuild that matrix's counts exactly."""
        start, end, stranded = self.of_type(flag)
        f, to = self.states(op["cell_flat"], start, end, stranded)
        n_states = op["P"].shape[0]
        counts = drifter_counts(self.drifters.traj_idx[start], f, to, n_states)
        C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
        assert np.array_equal(C, op["C"]), f"rebuilt counts differ from the saved {name} matrix"
        return counts
