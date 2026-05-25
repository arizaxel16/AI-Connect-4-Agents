"""
mcts_agent.py — Agente Connect-4 basado en Monte Carlo Tree Search (MCTS)
==========================================================================
Fundamento teórico (diapositivas de clase):
  - Pseudocódigo MCTS        → Slides 13, diapositiva 26
  - Fórmula UCB              → Slides 12, diapositiva 22
  - Trial-Based Online PImp  → Slides 13, diapositivas 11-21
  - Alternating Markov Game  → Slides 12, diapositivas 12-17

Idea central:
  En cada turno, el agente ejecuta N_SIMULATIONS simulaciones desde el
  estado actual. Cada simulación recorre el árbol de búsqueda usando UCB
  (para equilibrar exploración/explotación), expande un nodo nuevo,
  juega aleatoriamente hasta el final (rollout) y propaga el resultado
  hacia la raíz. La acción final es la columna con más visitas.

"""

# ── Dependencias ──────────────────────────────────────────────────────────────
import math
import numpy as np
from policy import Policy
from connect_state import ConnectState


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 1: MCTSNode — Un nodo dentro del árbol de búsqueda
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """
    Representa un nodo en el árbol MCTS.

    Cada nodo almacena:
      - El estado del tablero (ConnectState) en ese punto del juego.
      - Estadísticas de visitas (N) y victorias (W) para guiar la búsqueda.
      - Punteros al padre y a los hijos ya explorados.
      - La lista de acciones todavía no exploradas desde este nodo.
    """

    def __init__(self, state: ConnectState, parent=None, action: int = None):
        """
        Parámetros
        ----------
        state  : ConnectState — tablero y jugador activo en este nodo
        parent : MCTSNode     — nodo padre (None si es la raíz)
        action : int          — columna jugada para llegar a este nodo
        """
        # Estado del juego asociado a este nodo
        self.state = state

        # Estructura del árbol
        self.parent = parent          # Nodo del que venimos
        self.action = action          # Acción que nos trajo aquí

        # ── Estadísticas MCTS ────────────────────────────────────────────────
        # N: número de veces que este nodo fue visitado
        self.N = 0
        # W: número de victorias del jugador que ELIGIÓ este nodo
        #    (es decir, del jugador que estaba activo en el padre).
        #    Esto es lo que define el UCB clásico para juegos de suma cero.
        self.W = 0

        # ── Hijos y acciones pendientes ──────────────────────────────────────
        self.children = {}  # {accion (int): MCTSNode}

        # Acciones todavía no expandidas en este nodo.
        # Si el estado es terminal, la lista está vacía.
        if state.is_final():
            self.untried_actions = []
        else:
            # Mezclamos aleatoriamente para diversificar el orden de expansión
            free = state.get_free_cols()
            self.untried_actions = free[:]
            np.random.shuffle(self.untried_actions)

    # ── Propiedades de conveniencia ──────────────────────────────────────────

    @property
    def is_fully_expanded(self) -> bool:
        """True cuando todas las acciones posibles ya tienen hijo en el árbol."""
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        """True cuando el estado de juego es final (victoria o empate)."""
        return self.state.is_final()

    def ucb_score(self, C: float) -> float:
        """
        Calcula el puntaje UCB (Upper Confidence Bound) de este nodo,
        visto desde la perspectiva de su padre.

        Fórmula (Slides 12, diapositiva 22):
            UCB(s,a) = W/N  +  C · √(ln(N_padre) / N_hijo)

        El primer término (explotación) favorece nodos con alta tasa de victoria.
        El segundo término (exploración) favorece nodos poco visitados.
        """
        exploitation = self.W / self.N
        exploration  = C * math.sqrt(math.log(self.parent.N) / self.N)
        return exploitation + exploration


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 2: MCTSAgent — La política (agente) que usa MCTS
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSAgent(Policy):
    """
    Agente Connect-4 que implementa el algoritmo MCTS completo.

    Parámetros configurables
    ------------------------
    n_simulations : int   — presupuesto de simulaciones por turno
                            (más simulaciones = mejor juego, más tiempo)
    C             : float — constante de exploración UCB
                            (mayor C = más exploración, menor C = más explotación)

    Flujo de act():
      1. Componer el estado ConnectState desde el tablero recibido.
      2. Verificar jugadas tácticas inmediatas (ganar o bloquear).
      3. Ejecutar N simulaciones MCTS:
           SELECCIÓN → EXPANSIÓN → SIMULACIÓN → PROPAGACIÓN
      4. Retornar la columna del hijo con más visitas (política robusta).
    """

    def __init__(self, n_simulations: int = 500, C: float = 1.41):
        """
        n_simulations : presupuesto de simulaciones por turno.
        C             : constante de exploración UCB (√2 ≈ 1.41 es el valor teórico).
        """
        self.n_simulations = n_simulations
        self.C = C

    # ── Método obligatorio de Policy ────────────────────────────────────────

    def mount(self) -> None:
        """
        Llamado por el torneo antes de cada partida.
        MCTS no requiere entrenamiento offline, así que no hay nada que inicializar.
        (Ver Slides 13: el razonamiento ocurre completamente en línea, turno a turno.)
        """
        pass  # MCTS es 100 % online: no hay fase de entrenamiento.

    # ── Método principal ─────────────────────────────────────────────────────

    def act(self, board: np.ndarray) -> int:
        """
        Decide qué columna jugar dado el tablero actual.

        Parámetros
        ----------
        board : np.ndarray de shape (6, 7)
                -1 = ficha Roja, 1 = ficha Amarilla, 0 = celda vacía.

        Retorna
        -------
        int : columna elegida (0–6)
        """
        # ── Paso 1: Reconstruir el estado del juego ──────────────────────────
        current_player = self._infer_player(board)
        state = ConnectState(board, current_player)

        # ── Paso 2: Verificar jugadas tácticas inmediatas ────────────────────
        # Ganar o bloquear directamente sin necesidad de simulaciones.
        # Esto ahorra presupuesto para situaciones estratégicas más complejas.
        tactical_action = self._get_tactical_action(state, current_player)
        if tactical_action is not None:
            return tactical_action

        # ── Paso 3: Ejecutar MCTS ────────────────────────────────────────────
        root = MCTSNode(state)

        for _ in range(self.n_simulations):
            # (a) SELECCIÓN: bajar por el árbol usando UCB hasta un nodo hoja
            node = self._select(root)

            # (b) EXPANSIÓN: si no es terminal, agregar un hijo nuevo al árbol
            if not node.is_terminal:
                node = self._expand(node)

            # (c) SIMULACIÓN (rollout): jugar aleatoriamente hasta el final
            result = self._rollout(node.state)

            # (d) PROPAGACIÓN: actualizar N y W desde la hoja hasta la raíz
            self._backpropagate(node, result)

        # ── Paso 4: Elegir la acción más robusta ─────────────────────────────
        # Usamos la acción con MÁS VISITAS (no la mayor W/N).
        # La estrategia de "más visitas" es más robusta ante el ruido estadístico.
        best_action = max(root.children.keys(),
                          key=lambda a: root.children[a].N)
        return best_action

    # ═══════════════════════════════════════════════════════════════════════════
    # FASES DEL ALGORITMO MCTS
    # ═══════════════════════════════════════════════════════════════════════════

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        FASE 1 — SELECCIÓN (Slides 13, slide 26 línea 2)

        Baja por el árbol desde la raíz, eligiendo en cada nivel el hijo
        con el mayor puntaje UCB, hasta encontrar un nodo que:
          (a) todavía tenga acciones sin explorar (no está completamente expandido), o
          (b) sea un nodo terminal (fin de partida).
        """
        while not node.is_terminal and node.is_fully_expanded:
            node = self._best_ucb_child(node)
        return node

    def _best_ucb_child(self, node: MCTSNode) -> MCTSNode:
        """
        Elige el hijo con el puntaje UCB más alto.

        UCB = W/N + C·√(ln(N_padre)/N_hijo)
        (Fórmula textual de Slides 12, diapositiva 22)
        """
        return max(node.children.values(),
                   key=lambda child: child.ucb_score(self.C))

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        FASE 2 — EXPANSIÓN (Slides 13, slide 26 línea 3)

        Toma una acción no explorada, aplica la transición del juego
        y agrega el nodo hijo resultante al árbol.
        """
        # Sacar una acción no probada de la lista (ya está mezclada)
        action = node.untried_actions.pop()

        # Aplicar la transición para obtener el nuevo estado
        new_state = node.state.transition(action)

        # Crear el nodo hijo y conectarlo al árbol
        child = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child

        return child

    def _rollout(self, state: ConnectState) -> int:
        """
        FASE 3 — SIMULACIÓN / ROLLOUT (Slides 13, slide 26 línea 4)

        Juega la partida hasta el final usando una política aleatoria
        (llamada "default policy" en las diapositivas).

        Retorna el ganador: -1 (Rojo), 1 (Amarillo), 0 (empate).

        Por qué funciona: aunque los rollouts son "basura" individualmente,
        en promedio revelan qué acciones llevan a más victorias.
        (Explicación formal: Slides 13, diapositivas 17-21)
        """
        # Trabajamos con una copia para no modificar el estado del nodo
        current = state
        while not current.is_final():
            free_cols = current.get_free_cols()
            action = int(np.random.choice(free_cols))
            current = current.transition(action)
        return current.get_winner()

    def _backpropagate(self, node: MCTSNode, result: int) -> None:
        """
        FASE 4 — PROPAGACIÓN (Slides 13, slide 26 línea 6)

        Sube desde el nodo expandido hasta la raíz, actualizando
        las estadísticas N (visitas) y W (victorias) en cada nodo.

        Lógica de W:
          - Cada nodo almacena victorias del jugador que lo ELIGIÓ
            (es decir, del jugador que estaba activo en el padre).
          - Ese jugador es exactamente "-node.state.player"
            (porque node.state.player es quien va a mover DESDE este nodo,
             no quien llegó a él).
        """
        while node is not None:
            node.N += 1  # Una visita más a este nodo

            # El "dueño" de este nodo es quien movió para llegar aquí
            # = el jugador opuesto al que mueve desde este estado
            mover = -node.state.player

            if result == mover:
                # El mover ganó → suma una victoria
                node.W += 1
            # Si hay empate (result == 0) o perdió: W no cambia
            # Nota: no restamos en empate porque UCB usa W/N como estimación
            # de probabilidad de victoria, y un empate no es una victoria.

            node = node.parent  # Subir al padre

    # ═══════════════════════════════════════════════════════════════════════════
    # MÉTODOS AUXILIARES
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _infer_player(board: np.ndarray) -> int:
        """
        Determina a quién le toca mover a partir del tablero.

        Rojo (-1) siempre mueve primero. Si tiene el mismo número de fichas
        que Amarillo, es turno de Rojo. Si tiene una más, es turno de Amarillo.
        """
        red_count    = int(np.sum(board == -1))
        yellow_count = int(np.sum(board ==  1))
        return -1 if red_count == yellow_count else 1

    @staticmethod
    def _get_tactical_action(state: ConnectState, player: int):
        """
        Chequeo táctico ANTES de MCTS para dos situaciones críticas:

          1. ¿Puedo ganar ahora mismo? → Jugar inmediatamente.
          2. ¿Mi oponente gana en su próximo turno? → Bloquear.

        Esto garantiza que el agente no ignore victorias/bloqueos obvios
        incluso con un presupuesto de simulaciones muy bajo.

        Retorna la columna a jugar, o None si no hay jugada táctica urgente.
        """
        free_cols = state.get_free_cols()

        # ── Prioridad 1: ¿Puedo ganar? ──────────────────────────────────────
        for col in free_cols:
            next_state = state.transition(col)
            if next_state.get_winner() == player:
                return col  # ¡Victoria inmediata!

        # ── Prioridad 2: ¿Debo bloquear? ────────────────────────────────────
        opponent = -player
        for col in free_cols:
            # Simular que el oponente juega en esa columna
            opponent_state = ConnectState(state.board, opponent)
            if opponent_state.is_applicable(col):
                next_state = opponent_state.transition(col)
                if next_state.get_winner() == opponent:
                    return col  # Bloquear amenaza inmediata

        return None  # Sin jugada táctica urgente → usar MCTS
