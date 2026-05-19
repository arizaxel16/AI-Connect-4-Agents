"""
policy.py — Agente Connect-4: Política Greedy con Evaluación de Estado
=======================================================================
Concepto central (Slide 8 — Policy Improvement):
  Partimos de la política aleatoria y la mejoramos siendo greedy
  respecto a una función de evaluación Q̂(s, a) definida a mano:

      π(s) = argmax_a  Q̂(s, a)

  Q̂ asigna puntaje a cada acción según cuatro criterios en orden
  de prioridad: ganar > bloquear > amenazar > posición central.

  Esto es exactamente un paso de Policy Improvement sobre π_random.
"""

import random
try:
    from connect4.policy import Policy          # entorno real del torneo
except ModuleNotFoundError:
    from connect4_mock import Policy            # entorno local de pruebas


class GreedyAgent(Policy):
    """
    Agente greedy que evalúa cada columna legal y escoge la mejor.

    La función Q̂(s, a) suma cuatro componentes:

      +1 000  si la acción gana la partida inmediatamente
      +  500  si bloquea una victoria inmediata del oponente
      +   10  por cada secuencia de 3 propias en esa dirección
      +    1  por preferencia de columna central (0-3-0 simétrico)
    """

    # Peso de cada columna: el centro vale más que los bordes
    COL_SCORE = [0, 1, 2, 3, 2, 1, 0]

    def mount(self):
        pass

    def act(self, state) -> int:
        valid = [c for c in range(7) if state.is_applicable(c)]

        best_col   = valid[0]
        best_score = float("-inf")

        for col in valid:
            score = self._evaluate(state, col)
            if score > best_score:
                best_score = score
                best_col   = col

        return best_col

    # ── Evaluación ──────────────────────────────────────────────

    def _evaluate(self, state, col: int) -> float:
        """
        Calcula Q̂(state, col) para la acción 'col'.
        Cuatro componentes sumadas según prioridad.
        """
        board  = self._board(state)
        me     = self._my_player(board)
        opp    = 3 - me
        row    = self._landing_row(board, col)

        # ── Criterio 1: ¿Gano ahora? ───────────────────────────
        if self._wins(board, row, col, me):
            return 1_000

        # ── Criterio 2: ¿Bloqueo al oponente? ──────────────────
        if self._wins(board, row, col, opp):
            return 500

        # ── Criterio 3 + 4: amenazas propias + posición ─────────
        score  = self._threat_score(board, row, col, me)
        score += self.COL_SCORE[col]
        return score

    # ── Lógica del tablero ──────────────────────────────────────

    def _board(self, state) -> list:
        b = state.board
        return b.tolist() if hasattr(b, "tolist") else [list(r) for r in b]

    def _my_player(self, board) -> int:
        """Jugador 1 mueve primero. Si tienen igual fichas, soy el jugador 1."""
        n1 = sum(c == 1 for r in board for c in r)
        n2 = sum(c == 2 for r in board for c in r)
        return 1 if n1 == n2 else 2

    def _landing_row(self, board, col: int) -> int:
        """Fila donde caería la ficha en la columna dada."""
        for row in range(len(board) - 1, -1, -1):
            if board[row][col] == 0:
                return row
        return -1

    def _wins(self, board, row: int, col: int, player: int) -> bool:
        """¿Poner una ficha de 'player' en (row, col) forma 4 en raya?"""
        board[row][col] = player
        result = self._four_in_row(board, row, col, player)
        board[row][col] = 0
        return result

    def _four_in_row(self, board, row, col, player) -> bool:
        """Verifica las 4 direcciones centradas en (row, col)."""
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        R, C = len(board), len(board[0])
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + dr*sign, col + dc*sign
                while 0 <= r < R and 0 <= c < C and board[r][c] == player:
                    count += 1
                    r += dr*sign
                    c += dc*sign
            if count >= 4:
                return True
        return False

    def _threat_score(self, board, row: int, col: int, player: int) -> int:
        """
        Cuenta cuántas secuencias de 3 propias crea esta jugada.
        Cada secuencia suma 10 puntos (Policy Improvement: prefiero
        estados más cercanos a ganar).
        """
        board[row][col] = player
        score = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        R, C = len(board), len(board[0])
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + dr*sign, col + dc*sign
                while 0 <= r < R and 0 <= c < C and board[r][c] == player:
                    count += 1
                    r += dr*sign
                    c += dc*sign
            if count >= 3:
                score += 10
        board[row][col] = 0
        return score
