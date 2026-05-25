# Agente Connect-4 — MCTS (Grupo: Cali)

## Datos del grupo

| Campo | Detalle |
|---|---|
| **Nombre** | Juan Pablo Corral |
| **Algoritmo** | Monte Carlo Tree Search (MCTS) |
| **Archivo del agente** | `cali/agent/Policy.py` |
| **Clase del agente** | `MCTSAgent` |

---

## Requisitos

- Python 3.10+
- numpy
- matplotlib

Instalar dependencias:
```bash
pip install numpy matplotlib
```

---

## Guía de uso

### 1. Clonar el repositorio
```bash
git clone <URL-del-repositorio>
cd AI-Connect-4-Agents
```

### 2. Ejecutar el torneo
```bash
python main.py
```

### 3. Usar el agente directamente
```python
import numpy as np
from cali.agent.Policy import MCTSAgent

agente = MCTSAgent(n_simulations=500, C=1.41)
agente.mount()

tablero = np.zeros((6, 7), dtype=int)  # 0=vacío, -1=Rojo, 1=Amarillo
columna = agente.act(tablero)
print(f"El agente juega en la columna: {columna}")
```

### 4. Reproducir el análisis
```bash
cd cali/research
jupyter notebook entrega.ipynb
```
Ejecutar todas las celdas en orden. Las gráficas se guardan automáticamente como `.png`.

---

## Parámetros configurables

| Parámetro | Tipo | Descripción | Valor por defecto |
|---|---|---|---|
| `n_simulations` | `int` | Simulaciones por turno. Más simulaciones = mejor juego, más tiempo. | `500` |
| `C` | `float` | Constante de exploración UCB. Valor teórico óptimo: √2 ≈ 1.41 | `1.41` |

---

## Estructura de archivos relevantes

```
AI-Connect-4-Agents/
├── main.py                        ← punto de entrada del torneo
├── tournament/
│   └── connect4/
│       ├── connect_state.py       ← lógica del tablero
│       └── policy.py              ← clase base Policy
└── cali/
    ├── agent/
    │   └── Policy.py              ← MCTSAgent (agente entregado)
    ├── research/
    │   └── entrega.ipynb          ← análisis empírico con 6 gráficas
    └── README.md                  ← este archivo
```