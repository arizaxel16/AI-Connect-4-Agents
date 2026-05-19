"""
PiAgent.py — Agente Connect-4: Política Greedy con Evaluación de Estado
========================================================================
Concepto central (Slide 8 — Policy Improvement):
  Partimos de la política aleatoria y la mejoramos siendo greedy
  respecto a una función de evaluación Q-hat(s, a) definida a mano:

      pi(s) = argmax_a  Q-hat(s, a)

  Q-hat asigna puntaje a cada acción según cuatro criterios:
  ganar > bloquear > amenazar > posición central.

API del torneo:
  - act(s: np.ndarray) recibe el tablero como numpy array 6x7
  - Jugadores: -1 = Rojo (mueve primero), 1 = Amarillo, 0 = vacio
"""

import numpy as np
from policy import Policy


class GreedyAgent(Policy):
    """
    Agente greedy: evalua cada columna con Q-hat(s,a) y escoge la mejor.

      +1000  si la accion gana la partida inmediatamente
      + 500  si bloquea una victoria inmediata del oponente
      +  10  por cada secuencia de 3 propias que crea
      + 0-3  por preferencia de columna central
    """

    COL_SCORE = [0, 1, 2, 3, 2, 1, 0]

    def mount(self, *args, **kwargs) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        me    = self._current_player(s)
        opp   = -me
        valid = [c for c in range(7) if s[0, c] == 0]

        best_col, best_score = valid[0], float("-inf")
        for col in valid:
            score = self._evaluate(s, col, me, opp)
            if score > best_score:
                best_score, best_col = score, col
        return best_col

    def _evaluate(self, board, col, me, opp):
        row = self._landing_row(board, col)
        if self._wins(board, row, col, me):   return 1_000
        if self._wins(board, row, col, opp):  return 500
        return self._threat_score(board, row, col, me) + self.COL_SCORE[col]

    def _current_player(self, board):
        return -1 if (board == -1).sum() == (board == 1).sum() else 1

    def _landing_row(self, board, col):
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row
        return -1

    def _wins(self, board, row, col, player):
        b = board.copy()
        b[row, col] = player
        return self._four_in_row(b, row, col, player)

    def _four_in_row(self, board, row, col, player):
        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + dr*sign, col + dc*sign
                while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                    count += 1; r += dr*sign; c += dc*sign
            if count >= 4:
                return True
        return False

    def _threat_score(self, board, row, col, player):
        b = board.copy()
        b[row, col] = player
        score = 0
        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + dr*sign, col + dc*sign
                while 0 <= r < 6 and 0 <= c < 7 and b[r, c] == player:
                    count += 1; r += dr*sign; c += dc*sign
            if count >= 3:
                score += 10
        return score