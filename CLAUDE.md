# CLAUDE.md

## Linting & formato (OBLIGATORIO)

Este proyecto usa **Ruff** como formateador (equivalente a Prettier) y linter
(equivalente a ESLint). Config en [pyproject.toml](pyproject.toml).

**Siempre que hagas cambios en código Python, antes de terminar debes correr:**

```bash
.venv/bin/ruff format .        # formatea (como Prettier)
.venv/bin/ruff check --fix .   # lint + autofix (como ESLint --fix)
.venv/bin/ruff check .         # debe terminar en "All checks passed!"
```

Reglas:
- El árbol debe quedar en **"All checks passed!"** y sin reformateos pendientes
  (`ruff format --check .`) antes de dar una tarea por terminada.
- No introduzcas nuevas violaciones. Si una regla choca con un patrón deliberado,
  silénciala localmente con `# noqa: <CÓDIGO> (motivo)`, no desactivándola global.
- Tras correr el linter, ejecuta también los tests: `.venv/bin/python -m pytest -q`.

Las herramientas de desarrollo están en [requirements-dev.txt](requirements-dev.txt)
(`pip install -r requirements-dev.txt`).

## Estructura: stages como plugins

Cada stage del currículo es un **plugin autónomo** bajo
[src/stages/](src/stages/)`stageNN_<slug>/`:
- `plugin.py` — metadata del stage (número, nombre, gate, rehearsal, lr_scale,
  `trains_mood`, freeze-point, `enabled`) en un `StagePlugin`.
- `sources.py` — los generadores de datos PROPIOS del stage (un dict `SOURCES`:
  clave → builder).
- `data/level{L}/` — el corpus generado del stage vive DENTRO de su paquete
  (gitignored). Lo resuelve `src.stages.stage_data_dir`.

El **registry** ([src/stages/registry.py](src/stages/registry.py)) los descubre solos
(escanea `stageNN_*`), valida y responde todo lo de stages (`get_stage`,
`active_stages`, `bcf_stage`, `is_behavioral`, `mood_stages`, `stream_source`,
`stage_data_dir`). Helpers compartidos entre stages en `src/stages/_shared/`.

**Para añadir/quitar un stage:** suelta (o borra) un paquete `src/stages/stage11_*/`;
nada más se edita. Para desactivar uno: `enabled=False` en su plugin **o**
`curriculum.stageN.enabled: false` en el YAML del nivel (lo respetan train y prepare).

Otros puntos clave del refactor:
- Entrenador descompuesto en [src/training/](src/training/) (`trainer`, `gates`,
  `checkpoint`, `dataload`, `valdata`, `heads`, `curriculum`); CLI en
  [scripts/train.py](scripts/train.py). El daemon en
  [scripts/consolidation_daemon.py](scripts/consolidation_daemon.py).
- Núcleo de generación en [src/inference/generate.py](src/inference/generate.py)
  (lo reusan chat y agent).
- `src/training/stages.py` y `src/data/graded.py` quedan como **shims deprecados**
  que re-exportan desde las nuevas ubicaciones — no añadas código nuevo ahí.
