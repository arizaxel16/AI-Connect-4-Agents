"""
test_local.py — Ejecutar este script ANTES de subir a Gradescope
=================================================================
Cómo usarlo (desde la raíz del proyecto, donde está connect4/):
    python test_local.py

Si todo está bien verás:
    ✅ Import OK
    ✅ mount() OK
    ✅ act() devuelve columna válida: X
    ✅ Ganó Y de 10 partidas contra aleatorio
    ✅ LISTO PARA SUBIR

Si hay algún ❌ dime exactamente qué dice el error.
"""

import sys
import os

# ── Asegurar que el path apunta a la raíz del proyecto ───────────────────────
# Ajusta esta ruta si tu estructura es diferente
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np

# ── Test 1: Import ────────────────────────────────────────────────────────────
try:
    from mcts_agent import MCTSAgent
    print("✅ Import OK")
except Exception as e:
    print(f"❌ Error en import: {e}")
    sys.exit(1)

# ── Test 2: mount() ───────────────────────────────────────────────────────────
try:
    agent = MCTSAgent(n_simulations=50, C=1.41)
    agent.mount()
    print("✅ mount() OK")
except Exception as e:
    print(f"❌ Error en mount(): {e}")
    sys.exit(1)

# ── Test 3: act() con tablero vacío ──────────────────────────────────────────
try:
    board = np.zeros((6, 7), dtype=int)
    action = agent.act(board)
    assert isinstance(action, (int, np.integer)), f"act() debe retornar int, retornó {type(action)}"
    assert 0 <= action <= 6, f"Columna inválida: {action}"
    print(f"✅ act() devuelve columna válida: {action}")
except Exception as e:
    print(f"❌ Error en act(): {e}")
    sys.exit(1)

# ── Test 4: partida completa contra jugador aleatorio ────────────────────────
try:
    from connect_state import ConnectState

    wins = 0
    for game in range(10):
        state = ConnectState()
        agent2 = MCTSAgent(n_simulations=50)
        agent2.mount()

        while not state.is_final():
            if state.player == -1:           # nuestro agente juega como Rojo
                col = agent2.act(state.board)
            else:                            # aleatorio juega como Amarillo
                col = int(np.random.choice(state.get_free_cols()))

            if state.is_applicable(int(col)):
                state = state.transition(int(col))
            else:
                state = state.transition(state.get_free_cols()[0])

        if state.get_winner() == -1:
            wins += 1

    print(f"✅ Ganó {wins} de 10 partidas contra aleatorio (jugando como Rojo)")

except ImportError:
    # Si connect4 no está disponible, hacer la prueba con lógica interna
    from mcts_agent import _drop, _is_final, _winner, _free_cols

    wins = 0
    for game in range(10):
        board  = np.zeros((6,7), dtype=int)
        player = -1

        while not _is_final(board):
            if player == -1:
                col = agent.act(board)
            else:
                col = int(np.random.choice(_free_cols(board)))
            board  = _drop(board, int(col), player)
            player = -player

        if _winner(board) == -1:
            wins += 1

    print(f"✅ Ganó {wins} de 10 partidas contra aleatorio (lógica interna)")

except Exception as e:
    print(f"❌ Error en partida completa: {e}")
    sys.exit(1)

# ── Test 5: verificar herencia de Policy ─────────────────────────────────────
try:
    try:
        from policy import Policy
    except ImportError:
        from policy import Policy

    assert issubclass(MCTSAgent, Policy), "MCTSAgent debe heredar de Policy"
    print("✅ MCTSAgent hereda correctamente de Policy")
except Exception as e:
    print(f"❌ Error de herencia: {e}")
    sys.exit(1)

print("\n🏆 LISTO PARA SUBIR A GRADESCOPE")
