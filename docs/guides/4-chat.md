← [3-training.md](3-training.md) | → [5-eval.md](5-eval.md)

# Chat interactivo

Permite conversar con el modelo para evaluar coherencia, gramática,
razonamiento y comportamiento ético en EN y ES.

## Arrancar el chat

```bash
# Sin modelo entrenado — pipeline smoke test (output: token IDs random)
python chat.py --dummy

# Con checkpoint de una etapa entrenada
python chat.py --stage 1
python chat.py --stage 1 --lang es    # sesión en español
python chat.py --stage 1 --temp 0.5   # más conservador / predecible
python chat.py --stage 1 --temp 1.0   # más creativo / variado

# Checkpoint específico
python chat.py --checkpoint dist/checkpoints/stage3/final.npz
```

> No hace falta activar el venv. El script se auto-reinicia con `.venv/bin/python`
> si corrés `python chat.py` sin el entorno activado.

---

## Flags de línea de comandos

| Flag | Default | Descripción |
|---|---|---|
| `--stage N` | — | Carga el último checkpoint del Stage N |
| `--checkpoint PATH` | — | Carga un `.npz` específico |
| `--dummy` | — | Pesos random, sin checkpoint |
| `--lang en\|es` | `en` | Idioma inicial de la sesión |
| `--temp FLOAT` | `0.8` | Temperatura de muestreo |
| `--topp FLOAT` | `0.9` | Nucleus sampling p |
| `--maxtok INT` | `256` | Tokens máximos por respuesta |

---

## Comandos dentro del chat

```
/lang es        cambiar idioma de la sesión (en|es)
/temp 0.7       ajustar temperatura
/topp 0.9       ajustar nucleus sampling p
/maxtok 512     máximo tokens por respuesta
/stats          velocidad y parámetros de la última generación
/reset          borrar historial de contexto de la sesión
/quit           salir  (también Ctrl+C)
```

---

## Qué esperar según el estado del entrenamiento

| Estado | Output esperado |
|---|---|
| `--dummy` sin tokenizador | Token IDs numéricos (gibberish absoluto) |
| `--dummy` con tokenizador | Palabras reales pero completamente incoherentes |
| Stage 1 parcial (~500M tok) | Palabras frecuentes, frases sin sentido semántico |
| Stage 1 completo (gate pasado) | Frases gramaticalmente correctas, coherencia local |
| Stage 3+ | Responde preguntas matemáticas simples, analogías |
| Stage 5 completo | Resiste manipulación, responde dilemas éticos |

---

## Velocidad esperada en M2 Max

| Configuración | tok/s |
|---|---|
| Dummy / pesos random | 60–80 |
| Stage 1–5 (BF16) | 40–60 |
| Stage 5 + LoRA activo | 25–45 |
| Inferencia INT8 cuantizada | 60–90 |

Ver prompts de evaluación detallados en [6-eval.md](6-eval.md).
