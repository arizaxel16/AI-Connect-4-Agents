"""
John Doe v2 — depth-3 minimax with terminal-only scoring.

Decision rule:
    Run a minimax search of depth `self.depth` (default 3) over the game
    tree. The only scoring function at leaves is:
        +1  if the current board contains 4-in-a-row for me
        -1  if it contains 4-in-a-row for the opponent
         0  otherwise (quiet / unknown)
    Pick the move whose worst-case (opponent-best-play) outcome is highest.
    Break ties uniformly at random.

What depth 3 gains over the v1 1-step agent:
    1. Never plays a "suicide" move — any move that allows the opponent to
       win on their immediate reply gets a -1 score and is rejected unless
       *every* move is a loss.
    2. Sees and plays 1-move forks: if I have a move that produces two
       distinct winning threats, the opponent can only block one of them,
       so my reply on ply 3 wins → +1.

What depth 3 cannot guarantee:
    Forks created by the opponent that materialise on ply 4 of the search
    (i.e. opponent fork setup after my move). Those need depth >= 5.

The `depth` parameter is configurable for later analysis (depth-vs-winrate
sweep). The tournament harness always constructs with no args → depth=3.
"""

import numpy as np

from connect4.policy import Policy

ROWS = 6
COLS = 7

# Score sentinels. Using ±2 as ±infinity so the real scores ±1/0 compare
# correctly without importing math.inf.
WIN, LOSS, NEUTRAL = 1, -1, 0
NEG_INF, POS_INF = -2, 2


class JohnDoeV2(Policy):

    def __init__(self, depth: int = 3):
        self.depth = depth

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        my_color = self._infer_color(s)
        legal = [c for c in range(COLS) if s[0, c] == 0]

        # Fast path: any immediate winning move? Take it and skip the search.
        # (The search would also find it, but short-circuiting saves time.)
        for c in legal:
            board, r = self._drop(s.copy(), c, my_color)
            if self._has_four(board, r, c, my_color):
                return c

        # Score every legal move with minimax, remembering all moves that
        # tie for the best score so we can pick among them at random.
        best_score = NEG_INF
        best_moves: list[int] = []
        for c in legal:
            board, _ = self._drop(s.copy(), c, my_color)
            # After my move it's the opponent's turn, and we've used 1 ply
            # of the depth budget, so depth - 1 remains.
            score = self._minimax(board, -my_color, self.depth - 1, my_color)
            if score > best_score:
                best_score = score
                best_moves = [c]
            elif score == best_score:
                best_moves.append(c)

        return int(np.random.default_rng().choice(best_moves))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _minimax(
        self, board: np.ndarray, to_move: int, depth: int, my_color: int
    ) -> int:
        """Return the minimax value of `board` assuming `to_move` is next
        to play. Values are from MY perspective: +1 good, -1 bad, 0 neutral.

        Win-detection happens *before* the depth check, so we never miss a
        terminal node just because we ran out of depth budget on that ply.
        """
        legal = [c for c in range(COLS) if board[0, c] == 0]
        if not legal:
            return NEUTRAL  # board full, draw

        # If the player to move has an immediate winning column, they'll
        # take it. We can short-circuit here.
        for c in legal:
            test, r = self._drop(board.copy(), c, to_move)
            if self._has_four(test, r, c, to_move):
                return WIN if to_move == my_color else LOSS

        if depth == 0:
            return NEUTRAL  # depth exhausted, no terminal info

        if to_move == my_color:
            # MAX layer: I pick the move that maximizes my score.
            best = NEG_INF
            for c in legal:
                child, _ = self._drop(board.copy(), c, to_move)
                v = self._minimax(child, -to_move, depth - 1, my_color)
                if v > best:
                    best = v
                    if best == WIN:  # can't do better than a forced win
                        return WIN
            return best
        else:
            # MIN layer: opponent picks the move that minimizes my score.
            worst = POS_INF
            for c in legal:
                child, _ = self._drop(board.copy(), c, to_move)
                v = self._minimax(child, -to_move, depth - 1, my_color)
                if v < worst:
                    worst = v
                    if worst == LOSS:  # can't do worse than a forced loss
                        return LOSS
            return worst

    # ------------------------------------------------------------------
    # Board helpers (kept local so this file can be read on its own)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_color(s: np.ndarray) -> int:
        n_red = int(np.sum(s == -1))
        n_yellow = int(np.sum(s == 1))
        return -1 if n_red == n_yellow else 1

    @staticmethod
    def _drop(board: np.ndarray, col: int, color: int) -> tuple[np.ndarray, int]:
        """Mutate `board` in place — drop a `color` piece in `col`. Returns
        (board, landing_row). Caller passes a copy if they need to preserve
        the original."""
        for r in range(ROWS - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = color
                return board, r
        return board, -1

    @staticmethod
    def _has_four(board: np.ndarray, r: int, c: int, color: int) -> bool:
        """True iff a 4-in-a-row through (r, c) exists for `color`."""
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == color:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == color:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 4:
                return True
        return False
