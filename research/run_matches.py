"""
run_matches.py — Round-robin between the team's three Connect-4 agents.

Each pair (GreedyAgent, MCTSAgent, JohnDoeV2) plays 2N games: half with the
first agent as Red (moves first) and half with the second agent as Red.
Per-game results are written to results.csv.

Usage:
    python run_matches.py [N]        # N games per color per pair (default 10)
"""

import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import the tournament framework and each agent.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tournament"))           # exposes connect4.*
sys.path.insert(0, str(ROOT / "tournament" / "connect4"))  # bare imports inside

from connect_state import ConnectState  # noqa: E402


def _load(path: Path, class_name: str):
    """Load a single class from a file path without polluting sys.modules names."""
    spec = importlib.util.spec_from_file_location(f"_agent_{class_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


GreedyAgent = _load(ROOT / "pipe" / "agent" / "policy.py", "GreedyAgent")
MCTSAgent = _load(ROOT / "cali" / "agent" / "Policy.py", "MCTSAgent")
JohnDoeV2 = _load(ROOT / "axel" / "agent" / "john_doe_v2.py", "JohnDoeV2")

AGENTS = {
    "GreedyAgent": GreedyAgent,
    "MCTSAgent": MCTSAgent,
    "JohnDoeV2": JohnDoeV2,
}


def play_one(red_name: str, yellow_name: str, seed: int) -> dict:
    """Play one game. Red moves first (player = -1). Returns a result row."""
    np.random.seed(seed)  # makes MCTS / minimax tie-breaking reproducible
    red = AGENTS[red_name]()
    yellow = AGENTS[yellow_name]()
    red.mount()
    yellow.mount()

    state = ConnectState()  # player = -1 (Red) to move
    moves = 0
    red_time = 0.0
    yellow_time = 0.0

    while not state.is_final():
        current_name, current = (red_name, red) if state.player == -1 else (yellow_name, yellow)
        t0 = time.perf_counter()
        action = int(current.act(state.board))
        dt = time.perf_counter() - t0
        if state.player == -1:
            red_time += dt
        else:
            yellow_time += dt
        if not state.is_applicable(action):
            # Defensive: a misbehaving agent forfeits. Pick first legal column.
            action = state.get_free_cols()[0]
        state = state.transition(action)
        moves += 1

    winner = state.get_winner()
    if winner == -1:
        winner_name = red_name
    elif winner == 1:
        winner_name = yellow_name
    else:
        winner_name = "draw"

    return {
        "red": red_name,
        "yellow": yellow_name,
        "winner": winner_name,
        "moves": moves,
        "red_time_s": round(red_time, 4),
        "yellow_time_s": round(yellow_time, 4),
        "seed": seed,
    }


def main(n_per_color: int = 10) -> None:
    pairs = [
        ("GreedyAgent", "MCTSAgent"),
        ("GreedyAgent", "JohnDoeV2"),
        ("MCTSAgent", "JohnDoeV2"),
    ]
    out_path = Path(__file__).resolve().parent / "results.csv"

    rows: list[dict] = []
    seed = 1000
    for a, b in pairs:
        # a as Red, b as Yellow
        for i in range(n_per_color):
            print(f"[{a} (R) vs {b} (Y)]  game {i+1}/{n_per_color}", flush=True)
            rows.append(play_one(a, b, seed))
            seed += 1
        # swap colors so neither agent gets the first-move advantage
        for i in range(n_per_color):
            print(f"[{b} (R) vs {a} (Y)]  game {i+1}/{n_per_color}", flush=True)
            rows.append(play_one(b, a, seed))
            seed += 1

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
