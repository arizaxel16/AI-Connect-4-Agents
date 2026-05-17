"""
Evaluate JohnDoe vs a uniform-random opponent.

Plays N games, splits the report by who moved first so we can verify the
rubric prerequisite: never lose to random, win >= 50% as both colors.

Run from anywhere:
    python axel/agent/evaluate.py
"""

import sys
import pathlib
import numpy as np

# Standalone eval — we re-implement just enough of ConnectState locally so
# we don't drag in matplotlib/pydantic just to run a self-play loop.
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))                          # for john_doe
sys.path.insert(0, str(REPO_ROOT / "tournament"))      # for connect4.policy

from john_doe import JohnDoe, ROWS, COLS  # noqa: E402


class RandomPolicy:
    """Uniform-random over legal columns. Same behaviour as the placeholder
    policies under tournament/groups/. We don't inherit from the tournament
    Policy ABC here so the eval stays standalone — duck typing is enough."""

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        legal = [c for c in range(COLS) if s[0, c] == 0]
        return int(np.random.default_rng().choice(legal))


def _drop(board: np.ndarray, col: int, color: int) -> int:
    """Mutate `board` in place: drop a `color` piece in `col`. Return the
    row it landed in, or -1 if the column was full (shouldn't happen if
    callers pick from the legal list)."""
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = color
            return r
    return -1


def play_one_game(first, second) -> int:
    """Play one game. `first` plays Red (-1), `second` plays Yellow (+1).
    Returns -1, +1, or 0 (draw)."""
    first.mount()
    second.mount()
    board = np.zeros((ROWS, COLS), dtype=int)
    # We borrow JohnDoe's win-detector — it's pure and doesn't depend on
    # any agent state.
    has_four = JohnDoe._has_four
    to_move = -1  # Red moves first
    for _ in range(ROWS * COLS):
        current = first if to_move == -1 else second
        col = int(current.act(board))
        r = _drop(board, col, to_move)
        if has_four(board, r, col, to_move):
            return to_move
        to_move = -to_move
    return 0  # board full, no winner


def evaluate(n_games: int = 500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    # Tally John Doe's results split by the color it played.
    stats = {
        "as_red":    {"wins": 0, "losses": 0, "draws": 0},
        "as_yellow": {"wins": 0, "losses": 0, "draws": 0},
    }

    for _ in range(n_games):
        # Flip a fair coin: John Doe goes first (Red) or second (Yellow).
        john_is_red = rng.random() < 0.5
        if john_is_red:
            first, second = JohnDoe(), RandomPolicy()
            john_color = -1
            bucket = stats["as_red"]
        else:
            first, second = RandomPolicy(), JohnDoe()
            john_color = +1
            bucket = stats["as_yellow"]

        winner = play_one_game(first, second)
        if winner == 0:
            bucket["draws"] += 1
        elif winner == john_color:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1

    print(f"JohnDoe vs Random  —  {n_games} games\n")
    print(f"{'color':<12}{'wins':>6}{'losses':>8}{'draws':>7}{'win%':>8}")
    print("-" * 41)
    for color, b in stats.items():
        total = b["wins"] + b["losses"] + b["draws"]
        win_pct = 100.0 * b["wins"] / total if total else 0.0
        print(f"{color:<12}{b['wins']:>6}{b['losses']:>8}{b['draws']:>7}{win_pct:>7.1f}%")


if __name__ == "__main__":
    evaluate(n_games=500)
