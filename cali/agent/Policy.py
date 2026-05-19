import math
import numpy as np
from abc import ABC, abstractmethod

try:
    from connect4.policy import Policy
except ImportError:
    try:
        from policy import Policy
    except ImportError:
        class Policy(ABC):
            @abstractmethod
            def mount(self, *args, **kwargs) -> None:
                pass
            @abstractmethod
            def act(self, s) -> int:
                pass

try:
    from connect4.connect_state import ConnectState
except ImportError:
    from connect_state import ConnectState


# ── MCTSNode ──────────────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state  = state
        self.parent = parent
        self.action = action
        self.N = 0
        self.W = 0
        self.children = {}

        if state.is_final():
            self.untried_actions = []
        else:
            free = state.get_free_cols()
            self.untried_actions = free[:]
            np.random.shuffle(self.untried_actions)

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self):
        return self.state.is_final()

    def ucb_score(self, C):
        # UCB = W/N + C·√(ln(N_padre)/N_hijo)  — Slides 12, diap. 22
        return (self.W / self.N) + C * math.sqrt(math.log(self.parent.N) / self.N)


# ── MCTSAgent ─────────────────────────────────────────────────────────────────

class MCTSAgent(Policy):

    def __init__(self, n_simulations=500, C=1.41):
        self.n_simulations = n_simulations
        self.C = C

    def mount(self, *args, **kwargs) -> None:
        """
        Acepta el argumento de timeout que envía Gradescope (*args).
        MCTS no necesita inicialización — razona completamente en línea.
        """
        pass

    def act(self, board) -> int:
        board  = np.array(board, dtype=int)
        player = self._infer_player(board)
        state  = ConnectState(board, player)

        # ── GUARDIA 1: sin columnas libres → no hay jugada posible ───────────
        free_cols = state.get_free_cols()
        if not free_cols:
            return 0  # no debería ocurrir en una partida válida

        # ── GUARDIA 2: estado ya terminal → devolver columna aleatoria ────────
        # El test puede pasar tableros terminales; evitamos que root.children
        # quede vacío y max() explote con "empty sequence".
        if state.is_final():
            return int(np.random.choice(free_cols))

        # ── Chequeo táctico (ganar o bloquear antes de MCTS) ─────────────────
        tactic = self._get_tactical_action(state, player)
        if tactic is not None:
            return tactic

        # ── MCTS: construir árbol con N simulaciones ──────────────────────────
        root = MCTSNode(state)

        for _ in range(self.n_simulations):
            # (a) Selección: bajar con UCB hasta nodo no expandido
            node = self._select(root)

            # (b) Expansión: agregar hijo nuevo si no es terminal
            if not node.is_terminal:
                node = self._expand(node)

            # (c) Rollout: jugar aleatoriamente hasta el final
            result = self._rollout(node.state)

            # (d) Propagación: subir resultado actualizando N y W
            self._backpropagate(node, result)

        # ── GUARDIA 3: si por alguna razón no se crearon hijos → fallback ─────
        # Esto no debería ocurrir si el estado no es terminal, pero protege
        # contra cualquier condición inesperada del entorno de Gradescope.
        if not root.children:
            return int(np.random.choice(free_cols))

        # ── Decisión: columna con más visitas (más robusta que mayor W/N) ─────
        return max(root.children, key=lambda c: root.children[c].N)

    # ── Fases MCTS ────────────────────────────────────────────────────────────

    def _select(self, node):
        """Bajar por el árbol eligiendo siempre el hijo con mayor UCB."""
        while not node.is_terminal and node.is_fully_expanded:
            node = max(node.children.values(),
                       key=lambda ch: ch.ucb_score(self.C))
        return node

    def _expand(self, node):
        """Crear un nodo hijo para una acción no explorada."""
        action    = node.untried_actions.pop()
        new_state = node.state.transition(action)
        child     = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child
        return child

    def _rollout(self, state):
        """Jugar aleatoriamente hasta el fin y retornar el ganador."""
        current = state
        while not current.is_final():
            free    = current.get_free_cols()
            action  = int(np.random.choice(free))
            current = current.transition(action)
        return current.get_winner()

    def _backpropagate(self, node, result):
        """Subir actualizando N y W en cada nodo del camino."""
        while node is not None:
            node.N += 1
            # -node.state.player = quien eligió este nodo (jugador del padre)
            if result == -node.state.player:
                node.W += 1
            node = node.parent

    # ── Auxiliares ────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_player(board):
        """Rojo (-1) mueve primero. Igualdad de fichas → turno de Rojo."""
        return -1 if int(np.sum(board == -1)) == int(np.sum(board == 1)) else 1

    @staticmethod
    def _get_tactical_action(state, player):
        """
        Prioridad 1: victoria inmediata.
        Prioridad 2: bloqueo urgente.
        Siempre verificar is_applicable antes de transition.
        """
        free_cols = state.get_free_cols()
        opponent  = -player

        # ¿Puedo ganar ya?
        for col in free_cols:
            if not state.is_applicable(col):
                continue
            next_state = state.transition(col)
            if next_state.get_winner() == player:
                return col

        # ¿El oponente gana si no bloqueo?
        for col in free_cols:
            opp_state = ConnectState(state.board, opponent)
            if not opp_state.is_applicable(col):
                continue
            next_state = opp_state.transition(col)
            if next_state.get_winner() == opponent:
                return col

        return None