"""
test_agent.py — Test local del GreedyAgent
===========================================
Corre con:
    python test_agent.py

Requiere tener en el mismo directorio (o en el path):
    connect_state.py
    environment_state.py
    policy.py
    PiAgent.py

Jugadores: -1 = Rojo (mueve primero), 1 = Amarillo, 0 = vacío
"""

import sys, random
import numpy as np

from connect_state import ConnectState
from policy import Policy
from PiAgent import GreedyAgent


# ════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════

class RandomAgent(Policy):
    def mount(self) -> None:
        pass
    def act(self, s: np.ndarray) -> int:
        valid = [c for c in range(7) if s[0, c] == 0]
        return random.choice(valid)


def play_game(agent_red, agent_yellow) -> int:
    """
    Corre una partida completa.
    agent_red  juega como -1 (Rojo, mueve primero).
    agent_yellow juega como 1 (Amarillo).
    Retorna: -1 (gana Rojo), 1 (gana Amarillo), 0 (empate).
    """
    state = ConnectState()              # player=-1 por defecto
    while not state.is_final():
        if state.player == -1:
            col = agent_red.act(state.board)
        else:
            col = agent_yellow.act(state.board)
        state = state.transition(col)
    return state.get_winner()


def run_matches(agent_a, agent_b, n):
    """
    n partidas alternando quién juega de Rojo y Amarillo.
    Retorna (wins_a, wins_b, draws).
    """
    agent_a.mount(); agent_b.mount()
    wins_a = wins_b = draws = 0
    for i in range(n):
        if i % 2 == 0:
            r = play_game(agent_a, agent_b)   # a=Rojo, b=Amarillo
            if r == -1: wins_a += 1
            elif r ==1: wins_b += 1
            else:       draws  += 1
        else:
            r = play_game(agent_b, agent_a)   # b=Rojo, a=Amarillo
            if r == -1: wins_b += 1
            elif r ==1: wins_a += 1
            else:       draws  += 1
    return wins_a, wins_b, draws


def bar(value, total, width=30):
    filled = int(value / total * width) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_result(label, wins, losses, draws, total):
    wr = wins / total
    print(f"  {label}")
    print(f"    Victorias : {wins:3d}/{total}  {bar(wins, total)}  {wr:.1%}")
    print(f"    Derrotas  : {losses:3d}/{total}")
    print(f"    Empates   : {draws:3d}/{total}")
    print(f"    {'✅ Pasa' if wr > 0.50 else '❌ No pasa (necesita >50%)'}")
    print()


# ════════════════════════════════════════════════════════════════
#  SANITY CHECKS
# ════════════════════════════════════════════════════════════════

def test_wins_immediately():
    """Rojo (-1) tiene 3 en fila y debe ganar jugando en col 3."""
    board = np.array([
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [-1, -1, -1,  0,  1,  1,  0],
    ], dtype=int)
    col = GreedyAgent().act(board)
    ok  = (col == 3)
    print(f"  {'✅' if ok else '❌'} Detecta victoria inmediata → col {col} (esperado: 3)")
    return ok


def test_blocks_opponent():
    """Amarillo (1) tiene 3 en fila. Rojo (-1) debe bloquear en col 0 o 4."""
    board = np.array([
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  1,  1,  1,  0,  0,  0],
    ], dtype=int)
    # 0 fichas de -1 y 3 de 1 → turno de Rojo (-1) ✓
    col = GreedyAgent().act(board)
    ok  = col in (0, 4)
    print(f"  {'✅' if ok else '❌'} Bloquea al oponente → col {col} (esperado: 0 o 4)")
    return ok


def test_prefers_center():
    """En tablero vacío debe jugar en la columna central (3)."""
    board = np.zeros((6, 7), dtype=int)
    col   = GreedyAgent().act(board)
    ok    = (col == 3)
    print(f"  {'✅' if ok else '❌'} Prefiere el centro en tablero vacío → col {col} (esperado: 3)")
    return ok


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N = 100

    print("=" * 55)
    print("       TEST LOCAL — GreedyAgent Connect-4")
    print("       -1 = Rojo (primero)  |  1 = Amarillo")
    print("=" * 55)

    # ── Sanity checks ───────────────────────────────────────────
    print("\n── Sanity Checks ──────────────────────────────────────\n")
    ok1 = test_wins_immediately()
    ok2 = test_blocks_opponent()
    ok3 = test_prefers_center()
    print(f"\n  {'Todos pasaron ✓' if all([ok1,ok2,ok3]) else 'Algunos fallaron ⚠️'}\n")

    # ── vs Aleatorio como Rojo (-1) ─────────────────────────────
    print("── Test 1: GreedyAgent como Rojo (-1) vs Aleatorio ────\n")
    agent = GreedyAgent(); rnd = RandomAgent()
    agent.mount(); rnd.mount()
    wins = losses = draws = 0
    for _ in range(N):
        r = play_game(agent, rnd)
        if r == -1: wins   += 1
        elif r ==1: losses += 1
        else:       draws  += 1
    print_result("GreedyAgent como Rojo (-1)", wins, losses, draws, N)

    # ── vs Aleatorio como Amarillo (1) ──────────────────────────
    print("── Test 2: Aleatorio vs GreedyAgent como Amarillo (1) ─\n")
    agent.mount(); rnd.mount()
    wins = losses = draws = 0
    for _ in range(N):
        r = play_game(rnd, agent)
        if r ==  1: wins   += 1
        elif r ==-1: losses += 1
        else:        draws  += 1
    print_result("GreedyAgent como Amarillo (1)", wins, losses, draws, N)

    # ── Self-play ────────────────────────────────────────────────
    print("── Test 3: Self-play GreedyAgent vs GreedyAgent ───────\n")
    w1, w2, d = run_matches(GreedyAgent(), GreedyAgent(), N)
    print(f"  Instancia A gana: {w1}/{N}")
    print(f"  Instancia B gana: {w2}/{N}")
    print(f"  Empates         : {d}/{N}")
    print(f"  (Política determinista → mismo resultado siempre)\n")

    print("=" * 55)
    print("  Listo. Corre entrega.ipynb para las gráficas.")
    print("=" * 55)
