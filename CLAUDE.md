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

**Base a congelar:** cada stage DECLARA explícitamente si pertenece a la base
cognitiva que se congela, con `frozen_base: bool` en su `plugin.py` (True = cognitivo /
dentro de la base; False = behavioral / entrena un sector LoRA sobre el core ya
congelado). NO se decide por número de stage ni umbral — lo dice el propio stage.

**Tests por plugin:** cada stage mantiene SUS tests en
`src/stages/stageNN_<slug>/tests/`. Los tests transversales (registry, pipeline,
entrenamiento) viven en `tests/`. `pytest.ini` colecciona ambos (`testpaths =
tests src/stages`); el `conftest.py` de la raíz pone el repo en `sys.path`.

Otros puntos clave del refactor:
- Entrenador descompuesto en [src/training/](src/training/) (`trainer`, `gates`,
  `checkpoint`, `dataload`, `valdata`, `heads`, `curriculum`); CLI en
  [scripts/train.py](scripts/train.py). El daemon en
  `python -m src.consolidation.daemon` (src/consolidation/daemon.py).
- Núcleo de generación en [src/inference/generate.py](src/inference/generate.py)
  (lo reusan chat y agent).
- `src/training/stages.py` y `src/data/graded.py` quedan como **shims deprecados**
  que re-exportan desde las nuevas ubicaciones — no añadas código nuevo ahí.

## Convenciones de código (OBLIGATORIO)

**Nombres legibles.** Las variables deben referenciar su PROPÓSITO y leerse claras —
nada de una sola letra ni abreviaturas crípticas. Prefiere `stage_key` sobre `skey`,
`curriculum` sobre `cur`, `replay_weights` sobre `w`, `loader` sobre `ld`. Excepción:
índices triviales de loop (`i`) y convenciones matemáticas locales muy acotadas.

**Archivos pequeños, descompuestos.** Ningún archivo debe volverse "muy grande":
descompón en sub-archivos por responsabilidad (p. ej. el entrenador → `trainer` +
`gates` + `checkpoint` + `dataload` + `valdata` + `heads` + `curriculum`). Cuando
empieces a sentir un archivo pesado o con responsabilidades mezcladas, pártelo.

**Helpers reusables suben de nivel.** Si un helper lo usan ≥2 módulos, SÚBELO a una
carpeta común en vez de duplicarlo:
- compartido entre stages → `src/stages/_shared/`;
- compartido entre runtimes (chat/agent) → `src/inference/`;
- compartido en entrenamiento → un módulo en `src/training/`.
Nunca copies un helper en dos sitios — muévelo arriba y que ambos lo importen.

**Scripts vs internos.** En [scripts/](scripts/) van SOLO los CLIs accesibles para el
developer (train, prepare_data, train_tokenizer, run_benchmarks, plot_metrics, purge,
ood_probe, prepare_multimodal). Los componentes de runtime/internos NO van en scripts:
viven en su subsistema (p. ej. el daemon de consolidación en
`src/consolidation/daemon.py`, ejecutable con `python -m src.consolidation.daemon`).
