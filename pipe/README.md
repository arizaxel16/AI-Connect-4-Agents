# GreedyAgent — Connect-4

**Fundamentos de Inteligencia Artificial**  
Estrategia: Política Greedy con Evaluación de Estado (Policy Improvement)

---

## Idea principal

El agente implementa **un paso de Policy Improvement** sobre la política aleatoria (Slide 8).

Define una función de evaluación Q̂(s, a) para cada acción legal y elige siempre la acción con mayor puntaje:

```
π(s) = argmax_a  Q̂(s, a)
```

Q̂ evalúa cada columna con cuatro criterios en orden de prioridad:

| Prioridad | Criterio | Puntaje |
|---|---|---|
| 1 | Gano inmediatamente | +1 000 |
| 2 | Bloqueo victoria del oponente | +500 |
| 3 | Creo secuencia de 3 propias | +10 |
| 4 | Preferencia columna central | +0 a +3 |

---

## Comparación con los agentes del grupo

El grupo tiene tres agentes con enfoques conceptualmente distintos:

### Visión general

| Aspecto | **GreedyAgent** (Andrés / pipe) | **MCTSAgent** (Cali) | **JohnDoe v2** (Axel) |
|---|---|---|---|
| Concepto del curso | Policy Improvement greedy (Slide 8) | MCTS con UCB (Slide 12) | Minimax adversarial |
| Evaluación de estado | Q̂(s,a) con 4 criterios ponderados | Rollouts aleatorios → W/N | Terminal-only: ±1/0 (sin heurística) |
| Búsqueda en el árbol | **Ninguna** — evalúa 1 nivel | Árbol completo, 500 sims/turno | Árbol completo, profundidad 2 (def.) |
| Determinismo | **Totalmente determinista** | No-determinista (rollouts) | No-determinista (tie-breaking) |
| Parámetros | Ninguno (política fija) | `n_simulations`, `C` (UCB) | `depth` |
| Complejidad por turno | O(7) — instantáneo | O(n\_sim × d\_avg) | O(7^depth) exponencial |
| Aprendizaje | No requiere | Online (dentro del turno) | No requiere |

### Análisis por agente

**MCTSAgent (Cali)** — `cali/agent/Policy.py`  
Construye un árbol de búsqueda durante cada turno usando la fórmula UCB:

```
UCB = W/N + C · √(ln(N_padre) / N_hijo)
```

Con 500 simulaciones y C=1.41. Antes de MCTS aplica un chequeo táctico (victoria/bloqueo inmediato). Es el agente con **mayor capacidad de planeación** del grupo pero también el más costoso computacionalmente. Su evaluación es puramente estadística; no tiene conocimiento del dominio codificado.

**JohnDoe / JohnDoe v2 (Axel)** — `axel/agent/john_doe.py` / `john_doe_v2.py`  
- **v1**: win → block → random (1 ply, sin heurística de posición).  
- **v2**: minimax de profundidad `depth` con scoring terminal ±1. Evita "jugadas suicidas" (moves que dejan al oponente ganar en el siguiente turno) pero sin función heurística de evaluación intermedia — los nodos a profundidad máxima sin ganador se puntúan como 0.

**GreedyAgent (este agente)** — `pipe/agent/policy.py`  
La diferencia fundamental respecto a los otros dos es la **función Q̂(s,a) codificada con conocimiento del dominio**, que asigna puntajes directamente sin árbol ni rollouts. El orden de prioridad (ganar > bloquear > amenazar > centro) captura en O(1) lo que MCTS descubre estadísticamente y lo que Minimax explora de forma combinatoria. Es el único agente **completamente determinista** del grupo: dado el mismo tablero, siempre produce la misma jugada.

### Distinción conceptual clave

```
MCTS:        explora el espacio de estados con simulaciones aleatorias
Minimax:     busca exhaustivamente asumiendo oponente óptimo
GreedyAgent: evalúa directamente con conocimiento del dominio (sin búsqueda)
```

El GreedyAgent es el único que **no construye ninguna estructura de árbol** en tiempo de juego y cuya política es completamente fija y reproducible.

---

## Enlace al código

[`pipe/agent/policy.py` en branch `pipe`](https://github.com/arizaxel16/AI-Connect-4-Agents/blob/pipe/pipe/agent/policy.py)

---

## Estructura de archivos

```
pipe/
├── README.md              ← Este archivo
└── agent/
    ├── policy.py          ← Agente (GreedyAgent)
    ├── test_agent.py      ← Tests locales (sanity checks + win rate)
    └── entrega.ipynb      ← Análisis completo y gráficas
```

---

## Requisitos

```
python >= 3.10
numpy
matplotlib
```

El repositorio ya incluye el framework del torneo en `tournament/`. No se requiere instalación adicional.

---

## Cómo probar localmente

```bash
# Desde la raíz del repositorio
cd pipe/agent
python test_agent.py
```

Salida esperada: 3 sanity checks + win rates como Rojo y Amarillo vs aleatorio + self-play.

---

## Cómo ejecutar el análisis completo

```bash
# Desde la raíz del repositorio
jupyter notebook pipe/agent/entrega.ipynb
```

El notebook agrega `tournament/` al `sys.path` automáticamente. Ejecutar todas las celdas produce las tres gráficas (`exp1_win_rate_color.png`, `exp2_self_play.png`, `exp3_ablacion.png`).

---

## Cómo usarlo en el torneo

```python
from connect4.policy import Policy
# El agente vive en pipe/agent/policy.py
# El torneo lo carga desde tournament/groups/

agent = GreedyAgent()
agent.mount()
col = agent.act(state)   # retorna columna 0–6
```

No tiene parámetros configurables: la política es determinista.

---

## Concepto del curso aplicado

**Slide 8 — Policy Improvement:**

> Dada una política π, se puede construir una política π' ≥ π siendo greedy respecto al valor de π.

Aquí π es la política aleatoria y π' = GreedyAgent.  
La función Q̂(s,a) es la "evaluación" de cada acción — no aprendida, sino definida por conocimiento del dominio.
