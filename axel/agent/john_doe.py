"""
John Doe — the baseline 1-step-lookahead Connect-4 agent.

Decision rule, in priority order, every turn:
    1. If there is a column where dropping my piece wins immediately, play it.
    2. Else if there is a column where the opponent would win on their next
       turn, play it (block).
    3. Otherwise, play a uniformly random legal column.

No training, no search beyond one ply, no persistent state between games.
This agent exists to (a) verify the harness wiring end-to-end and
(b) serve as a baseline that every smarter agent must comfortably beat.
"""

import numpy as np

from connect4.policy import Policy


ROWS = 6
COLS = 7


class JohnDoe(Policy):

    def mount(self) -> None:
        # Nothing to set up — no model to load, no tables to build.
        # A fresh instance is constructed for every game by the tournament
        # harness, so per-game state would go here if we needed it.
        pass

    def act(self, s: np.ndarray) -> int:
        my_color = self._infer_color(s)
        opp_color = -my_color
        legal = [c for c in range(COLS) if s[0, c] == 0]

        # Rule 1: take an immediate win if one exists.
        for c in legal:
            if self._would_win(s, c, my_color):
                return c

        # Rule 2: block an immediate opponent win if one exists.
        for c in legal:
            if self._would_win(s, c, opp_color):
                return c

        # Rule 3: random legal fallback.
        return int(np.random.default_rng().choice(legal))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_color(s: np.ndarray) -> int:
        """
        Figure out whose turn it is from the board alone.

        Red (-1) always moves first, so after an even number of total moves
        it is Red's turn; after an odd number it is Yellow's.
        """
        n_red = int(np.sum(s == -1))
        n_yellow = int(np.sum(s == 1))
        return -1 if n_red == n_yellow else 1

    @staticmethod
    def _drop_row(s: np.ndarray, col: int) -> int:
        """Return the row index a piece would land on if dropped in `col`,
        or -1 if the column is full."""
        for r in range(ROWS - 1, -1, -1):
            if s[r, col] == 0:
                return r
        return -1

    def _would_win(self, s: np.ndarray, col: int, color: int) -> bool:
        """Simulate dropping a `color` piece in `col` and return True iff
        that move produces a 4-in-a-row for `color`."""
        r = self._drop_row(s, col)
        if r == -1:
            return False
        # Only check lines through the new cell — that's the only thing
        # that could have changed.
        return self._has_four(s, r, col, color)

    @staticmethod
    def _has_four(s: np.ndarray, r: int, c: int, color: int) -> bool:
        """Return True iff placing `color` at (r, c) creates 4-in-a-row
        through that cell. The cell (r, c) is treated as already containing
        `color` for the check, even if `s[r, c]` is still 0."""
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]
        for dr, dc in directions:
            count = 1  # the new piece itself
            # Walk forward along (dr, dc)
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and s[rr, cc] == color:
                count += 1
                rr += dr
                cc += dc
            # Walk backward along (-dr, -dc)
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and s[rr, cc] == color:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 4:
                return True
        return False
