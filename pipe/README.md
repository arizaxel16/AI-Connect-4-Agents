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

## Diferencia respecto al compañero (MCTS)

| Aspecto | MCTS (compañero) | GreedyAgent (este) |
|---|---|---|
| Simulaciones | Corre cientos de partidas por turno | No corre ninguna |
| Árbol de búsqueda | Construye árbol completo | No construye árbol |
| Evaluación | Estadística (wins/visits) | Función directa Q̂(s,a) |
| Concepto del curso | Online Policy Improvement | Policy Improvement greedy |
| Velocidad | Más lento | Instantáneo |

---

## Estructura de archivos

```
grupos/grupo_greedy/
├── policy.py          ← Agente (GreedyAgent)
├── connect4_mock.py   ← Simulador local (solo para pruebas)
├── test_agent.py      ← Tests locales
└── README.md          ← Este archivo

entrega.ipynb          ← Análisis y gráficas
```

---

## Cómo probar localmente

```bash
# Desde la carpeta del agente
python test_agent.py
```

Salida esperada: 3 sanity checks + win rates contra el aleatorio.

---

## Cómo usarlo en el torneo

```python
from groups.grupo_greedy.policy import GreedyAgent

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
