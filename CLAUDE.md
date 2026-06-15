# CLAUDE.md

## Cero tech-debt (OBLIGATORIO)

**Tolerancia 0 a deuda técnica.** Todo lo que no se use o no sirva se ELIMINA, no se
deja "por si acaso":
- Nada de shims deprecados, aliases "back-compat", re-exports para módulos que ya no
  existen, parámetros muertos, ramas inalcanzables, ni código comentado. Si algo queda
  sin usar tras un cambio, BÓRRALO en el mismo cambio (incluido su test si solo existía
  para cubrir el alias/shim).
- Una sola fuente de verdad por concepto. Si encuentras dos formas de hacer lo mismo,
  unifica y elimina la otra; nunca dupliques un helper (súbelo, ver más abajo).
- Migra a los consumidores en el mismo PR cuando renombres/muevas algo, y borra el
  nombre viejo — no lo mantengas vivo con un alias.
- Antes de dar una tarea por terminada: `grep` de nombres/rutas viejas debe quedar
  VACÍO, y no debe haber imports/símbolos sin usar (Ruff F401/F811 lo detecta).

Al revisar o tocar un área, deja también limpio lo que encuentres alrededor
(boy-scout rule): incongruencias de nombres, semántica engañosa, y muertos.

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

## Arquitectura de carpetas: `src/` (framework) vs `models/` (OBLIGATORIO)

Separación de nivel raíz: **`src/` = SOLO el framework; `models/` = los modelos** (fuera
de `src/`). El framework NUNCA importa un modelo (lo descubre por nombre vía el registry);
un modelo SÍ consume el framework.

- **`src/`** — el FRAMEWORK, **agnóstico a la tarea**: todo lo general
  (`backend/`, `model/`, `modalities/`, `data/`, `training/`, `consolidation/`,
  `memory/`, `relevance/`, `routing/` + `config.py`, `env.py`, `resources.py`,
  `observability.py`). Cambiando el modelo, este mismo framework entrena OTRO tipo de
  modelo. **El framework NUNCA importa un modelo/stage concreto** ni importa de un `uses/`.
- **`src/plugins/`** — el SISTEMA de plugins (framework, NO es un plugin): el contrato
  (`base.py` → `StagePlugin`/`ModelSpec`), el `registry.py` que descubre los modelos en
  `models/`, y el **SDK** ([src/plugins/sdk/](src/plugins/sdk/)) — el único import estable
  del que dependen los stages.
- **`models/`** — los MODELOS, fuera de `src/`. Cada uno (`models/<nombre>/`) es un
  escenario autónomo: sus `stageNN_*`, sus faculties propias (p. ej. `mood/` en cognition),
  sus **`uses/`** (consumidores: chat, agent…) y sus `experiments/`. Un stage solo se
  CONSUME y depende ÚNICAMENTE del SDK (`src/plugins/sdk`); helpers específicos de su
  modelo los importa intra-modelo. Borrar `models/<nombre>/` lo quita entero sin tocar el
  framework.
- **`models/<nombre>/uses/`** — los CONSUMIDORES de ESE modelo (chat, agent, futura API).
  NO son framework: son formas de consumir el modelo ya creado, AHORA por-modelo. Lo
  compartido entre consumidores del modelo va en `models/<nombre>/uses/common/`
  (`agent.py`, `generate.py`, `loading.py`, `interaction.py`). Un `uses/` puede importar de
  `src` y de su propio modelo; el framework NUNCA importa de un `uses/`.

## Modelos: un framework, varios escenarios

Un **modelo** es un escenario de entrenamiento bajo `models/<nombre>/` y agrupa sus
propios stages (cada modelo internamente corre un grupo de stages):
- **`models/cognition/`** — el LLM conversacional/agéntico (los 10 stages actuales).
  Es el modelo **por defecto** (`cfg["model_name"]` ausente → `cognition`).
- **`models/hands_recognition/`** — TODO (pose de manos para VR), el segundo modelo
  que demuestra que el framework es agnóstico: solo es un stub con instrucciones.

Cada modelo tiene además sus **experiments** propios (sondas de hipótesis) en
`models/<nombre>/experiments/` — p. ej.
[models/cognition/experiments/continual_learning.py](models/cognition/experiments/continual_learning.py).

El modelo activo se elige con `cfg["model_name"]` (clave del YAML del nivel; OJO: distinta
de `model:`, que es la ARQUITECTURA — d_model, n_layers — de ese modelo). `scripts/train.py`
y `scripts/prepare_data.py` llaman `set_active_model(cfg.get("model_name"))` al arrancar,
ANTES de tocar el registry, para que descubra los stages de ESE modelo.

**`ModelSpec`** ([src/plugins/base.py](src/plugins/base.py)) es la costura que hace el motor
agnóstico a la tarea/modalidad. El trainer NUNCA construye la red, el loader, la pérdida ni
el gate directamente: se los pide al `ModelSpec` activo (resuelto en
[src/training/model_spec.py](src/training/model_spec.py)). El spec por defecto
(`text-lm`) cablea las piezas del LLM de texto (`setup.build_stage_model`,
`dataload.build_data_loader`, la objetivo MRL+aux, `gates.evaluate_gate`), así `cognition`
se comporta idéntico que antes. Un modelo sobreescribe cualquier pieza exponiendo
`SPEC = ModelSpec(...)` (o `build_spec(cfg) -> ModelSpec`) en su paquete. Convención: en
`evaluate`, **menor score es mejor** (una métrica higher-is-better devuelve p. ej.
`1-accuracy`), para que el ratchet del trainer siga siendo agnóstico a la métrica.

**Código específico del modelo vive CON el modelo.** El framework solo contiene lo
REUSABLE/general del framework; nada atado a un modelo concreto. Si una faculty solo
tiene sentido para un modelo (p. ej. los **moods/emociones** son de cognition — no
sirven a un detector de manos), va en `models/<modelo>/` (ver
[models/cognition/mood/](models/cognition/mood/)), NO en `src` ni en el SDK.
- El framework NUNCA importa el modelo. Para efectos secundarios específicos del modelo al
  terminar un stage (p. ej. entrenar el mood head de cognition), el trainer invoca un
  **hook opcional** `post_stage(model, stage, cfg, ckpt_dir, precision)` que el paquete
  del modelo expone — descubierto por nombre vía `model_hook(...)` (mismo patrón que
  `SPEC`). Así el core agnóstico dispara trabajo del modelo sin importarlo.
- Un **stage** importa helpers GENERALES del framework solo desde el SDK; los helpers
  ESPECÍFICOS de su modelo los importa intra-modelo (de `models/<modelo>/…`), nunca
  de otro modelo ni de `src`. Borrar el modelo se lleva su faculty y sus stages.
- `uses/` (consumidores) importa la faculty del modelo directamente (p. ej.
  `from models.cognition.mood import MoodTracker`). El SDK permanece libre de mood.

## Estructura: stages como plugins

Cada stage del currículo es un **plugin autónomo** bajo
`models/<modelo>/stageNN_<slug>/`:
- `plugin.py` — metadata del stage (número, nombre, gate, rehearsal, lr_scale,
  `trains_mood`, freeze-point, `enabled`) en un `StagePlugin`.
- `sources.py` — los generadores de datos PROPIOS del stage (un dict `SOURCES`:
  clave → builder).
- `data/level{L}/` — el corpus generado del stage vive DENTRO de su paquete
  (gitignored). Lo resuelve `src.plugins.stage_data_dir`.

El **registry** ([src/plugins/registry.py](src/plugins/registry.py)) los descubre solos
(escanea `stageNN_*` del modelo activo), valida y responde todo lo de stages (`get_stage`,
`active_stages`, `bcf_stage`, `is_behavioral`, `mood_stages`, `stream_source`,
`stage_data_dir`). `bcf_stage()` devuelve `int | None` (un modelo puede NO tener punto de
congelado → freeze opcional). Un stage es autónomo: su ÚNICO acoplamiento al framework es
`from src.plugins.sdk import ...` (contrato + helpers: `StagePlugin`, `StageGate`,
`blend`, `interleave`, `cycle_records`, `stable_hash`, `passes_filter`, `persona_for`,
`prepend_system`, `hermes_events`, `emotion_to_mood`, …). Helpers usados solo por UN
stage viven dentro de su propio paquete (p. ej.
`models/cognition/stage01_language/dictionary.py`). El SDK es el puente al core; el
plugin nunca pasa de él.

**Para añadir/quitar un stage:** suelta (o borra) un paquete
`models/<modelo>/stage11_*/`; nada más se edita. Para desactivar uno: `enabled=False`
en su plugin **o** `curriculum.stageN.enabled: false` en el YAML del nivel (lo respetan
train y prepare).

**Base a congelar:** cada stage DECLARA explícitamente si pertenece a la base
cognitiva que se congela, con `frozen_base: bool` en su `plugin.py` (True = cognitivo /
dentro de la base; False = behavioral / entrena un sector LoRA sobre el core ya
congelado). NO se decide por número de stage ni umbral — lo dice el propio stage.

**Tests por plugin:** cada stage mantiene SUS tests en
`models/<modelo>/stageNN_<slug>/tests/` (borrar el stage se lleva sus tests). Los
tests transversales (registry, pipeline, entrenamiento, y los del framework/SDK como
loader y `blend`) viven en `tests/` y NO importan ningún stage concreto. `pytest.ini`
colecciona ambos (`testpaths = tests models`); el `conftest.py` de la raíz pone el
repo en `sys.path`.

Otros puntos clave del refactor:
- Entrenador descompuesto en [src/training/](src/training/) (`trainer`,
  `setup`, `gates`, `checkpoint`, `dataload`, `valdata`, `heads`, `curriculum`); CLI en
  [scripts/train.py](scripts/train.py). El daemon en
  `python -m src.consolidation.daemon` (src/consolidation/daemon.py).
- Núcleo de generación en [models/cognition/uses/common/generate.py](models/cognition/uses/common/generate.py) y carga de
  modelo/checkpoint en [models/cognition/uses/common/loading.py](models/cognition/uses/common/loading.py) — son
  CONSUMIDORES (los reusan chat y agent), por eso viven en `uses/`, no en el framework.
- La metadata de stages (gates, nombres, rehearsal, lr_scale, freeze point, mood) vive
  SOLO en los plugins y la sirve `src.plugins` (registry) — no hay tablas duplicadas. (Los
  antiguos `src/data/graded.py` y el shim `src/training/stages.py` se ELIMINARON.)

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
- compartido entre plugins/stages → el **SDK de plugins** (`src/plugins/sdk/`), que es el
  único contrato del que dependen los plugins; **nunca** un `_shared` dentro de `plugins/`
  ni un import directo de `src` desde un plugin;
- compartido entre consumidores (chat/agent) → `models/cognition/uses/common/`;
- compartido en el framework agnóstico → su subsistema en `src/` (entrenamiento →
  `src/training/`, etc.).
Nunca copies un helper en dos sitios — muévelo arriba y que ambos lo importen.

**CLI único: `rdmca`.** [scripts/rdmca.py](scripts/rdmca.py) es el ÚNICO punto de
entrada para el developer: agrupa todo (`prepare`, `tokenizer`, `prepare-mm`, `train`,
`bench`, `ood`, `plot`, `chat`, `agent`, `daemon`, `purge`) y reenvía los args a la
herramienta real (así `rdmca train --help` muestra los args verdaderos — una sola fuente
de verdad por comando, sin duplicar argparse). `rdmca info [--model M] [--level L]` es
model-aware: lista modelos, niveles y stages, y marca qué está preparado/entrenado.
Selección de modelo con `--model` (override de `cfg["model_name"]`) en los comandos que
tocan stages.

**Scripts vs internos.** Los scripts en [scripts/](scripts/) son los CLIs del developer
(envueltos por `rdmca`). Los componentes de runtime/internos NO van en scripts: viven en
su subsistema (p. ej. el daemon de consolidación en `src/consolidation/daemon.py`,
ejecutable con `rdmca daemon` o `python -m src.consolidation.daemon`). Las apps que
CONSUMEN el modelo (chat, agent) viven en `uses/`, no en `scripts/`. Al añadir un CLI
nuevo, regístralo en `COMMANDS` de `rdmca.py`.
