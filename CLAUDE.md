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

## Arquitectura de carpetas: core · models · uses (OBLIGATORIO)

Tres capas, separadas a propósito:

- **`src/core/`** — el FRAMEWORK, **agnóstico a la tarea**: todo lo general
  (`backend/`, `model/`, `modalities/`, `data/`, `training/`, `consolidation/`,
  `memory/`, `relevance/`, `routing/` + `config.py`, `env.py`, `resources.py`,
  `observability.py`). El objetivo es que, cambiando los modelos/plugins, este mismo
  framework pueda entrenar OTRO tipo de modelo (p. ej. reconocimiento de manos para VR).
  **El core NUNCA importa un stage concreto** (solo descubre los plugins vía el registry)
  ni importa de `uses/`.
- **`src/models/`** — los modelos, cada uno un escenario de entrenamiento con sus stages,
  **100% aislados**: un stage solo se CONSUME, no depende de nada del framework salvo del
  **SDK de plugins** ([src/models/sdk/](src/models/sdk/)) — un único import estable. NUNCA
  importa de `src/core` directamente, ni de otros stages. Así borrar un stage (`rm -rf` su
  carpeta) jamás rompe el framework ni otro stage, y añadir uno solo requiere el SDK.
  El propio sistema de plugins (`base.py`, `registry.py`, `sdk/`) vive aquí pero NO es
  un plugin.
- **`uses/`** — los CONSUMIDORES del modelo (chat, agent, futura API). NO son parte del
  framework: son formas de consumir el modelo ya creado. Lo compartido entre
  consumidores va en `uses/common/` (`agent.py`, `generate.py`, `loading.py`,
  `interaction.py`). `uses/` puede importar de `src/core` y `src/models`; el framework
  NUNCA importa de `uses/`.

## Modelos: un framework, varios escenarios

Un **modelo** es un escenario de entrenamiento bajo `src/models/<nombre>/` y agrupa sus
propios stages (cada modelo internamente corre un grupo de stages):
- **`src/models/cognition/`** — el LLM conversacional/agéntico (los 10 stages actuales).
  Es el modelo **por defecto** (`cfg["model_name"]` ausente → `cognition`).
- **`src/models/hands_recognition/`** — TODO (pose de manos para VR), el segundo modelo
  que demuestra que el framework es agnóstico: solo es un stub con instrucciones.

Cada modelo tiene además sus **experiments** propios (sondas de hipótesis) en
`src/models/<nombre>/experiments/` — p. ej.
[src/models/cognition/experiments/continual_learning.py](src/models/cognition/experiments/continual_learning.py).

El modelo activo se elige con `cfg["model_name"]` (clave del YAML del nivel; OJO: distinta
de `model:`, que es la ARQUITECTURA — d_model, n_layers — de ese modelo). `scripts/train.py`
y `scripts/prepare_data.py` llaman `set_active_model(cfg.get("model_name"))` al arrancar,
ANTES de tocar el registry, para que descubra los stages de ESE modelo.

**`ModelSpec`** ([src/models/base.py](src/models/base.py)) es la costura que hace el motor
agnóstico a la tarea/modalidad. El trainer NUNCA construye la red, el loader, la pérdida ni
el gate directamente: se los pide al `ModelSpec` activo (resuelto en
[src/core/training/model_spec.py](src/core/training/model_spec.py)). El spec por defecto
(`text-lm`) cablea las piezas del LLM de texto (`setup.build_stage_model`,
`dataload.build_data_loader`, la objetivo MRL+aux, `gates.evaluate_gate`), así `cognition`
se comporta idéntico que antes. Un modelo sobreescribe cualquier pieza exponiendo
`SPEC = ModelSpec(...)` (o `build_spec(cfg) -> ModelSpec`) en su paquete. Convención: en
`evaluate`, **menor score es mejor** (una métrica higher-is-better devuelve p. ej.
`1-accuracy`), para que el ratchet del trainer siga siendo agnóstico a la métrica.

## Estructura: stages como plugins

Cada stage del currículo es un **plugin autónomo** bajo
`src/models/<modelo>/stageNN_<slug>/`:
- `plugin.py` — metadata del stage (número, nombre, gate, rehearsal, lr_scale,
  `trains_mood`, freeze-point, `enabled`) en un `StagePlugin`.
- `sources.py` — los generadores de datos PROPIOS del stage (un dict `SOURCES`:
  clave → builder).
- `data/level{L}/` — el corpus generado del stage vive DENTRO de su paquete
  (gitignored). Lo resuelve `src.models.stage_data_dir`.

El **registry** ([src/models/registry.py](src/models/registry.py)) los descubre solos
(escanea `stageNN_*` del modelo activo), valida y responde todo lo de stages (`get_stage`,
`active_stages`, `bcf_stage`, `is_behavioral`, `mood_stages`, `stream_source`,
`stage_data_dir`). `bcf_stage()` devuelve `int | None` (un modelo puede NO tener punto de
congelado → freeze opcional). Un stage es autónomo: su ÚNICO acoplamiento al framework es
`from src.models.sdk import ...` (contrato + helpers: `StagePlugin`, `StageGate`,
`blend`, `interleave`, `cycle_records`, `stable_hash`, `passes_filter`, `persona_for`,
`prepend_system`, `hermes_events`, `emotion_to_mood`, …). Helpers usados solo por UN
stage viven dentro de su propio paquete (p. ej.
`src/models/cognition/stage01_language/dictionary.py`). El SDK es el puente al core; el
plugin nunca pasa de él.

**Para añadir/quitar un stage:** suelta (o borra) un paquete
`src/models/<modelo>/stage11_*/`; nada más se edita. Para desactivar uno: `enabled=False`
en su plugin **o** `curriculum.stageN.enabled: false` en el YAML del nivel (lo respetan
train y prepare).

**Base a congelar:** cada stage DECLARA explícitamente si pertenece a la base
cognitiva que se congela, con `frozen_base: bool` en su `plugin.py` (True = cognitivo /
dentro de la base; False = behavioral / entrena un sector LoRA sobre el core ya
congelado). NO se decide por número de stage ni umbral — lo dice el propio stage.

**Tests por plugin:** cada stage mantiene SUS tests en
`src/models/<modelo>/stageNN_<slug>/tests/` (borrar el stage se lleva sus tests). Los
tests transversales (registry, pipeline, entrenamiento, y los del framework/SDK como
loader y `blend`) viven en `tests/` y NO importan ningún stage concreto. `pytest.ini`
colecciona ambos (`testpaths = tests src/models`); el `conftest.py` de la raíz pone el
repo en `sys.path`.

Otros puntos clave del refactor:
- Entrenador descompuesto en [src/core/training/](src/core/training/) (`trainer`,
  `setup`, `gates`, `checkpoint`, `dataload`, `valdata`, `heads`, `curriculum`); CLI en
  [scripts/train.py](scripts/train.py). El daemon en
  `python -m src.core.consolidation.daemon` (src/core/consolidation/daemon.py).
- Núcleo de generación en [uses/common/generate.py](uses/common/generate.py) y carga de
  modelo/checkpoint en [uses/common/loading.py](uses/common/loading.py) — son
  CONSUMIDORES (los reusan chat y agent), por eso viven en `uses/`, no en el framework.
- La metadata de stages (gates, nombres, rehearsal, lr_scale, freeze point, mood) vive
  SOLO en los plugins y la sirve `src.models` (registry) — no hay tablas duplicadas. (Los
  antiguos `src/data/graded.py` y el shim `src/core/training/stages.py` se ELIMINARON.)

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
- compartido entre plugins/stages → el **SDK de plugins** (`src/models/sdk/`), que es el
  único contrato del que dependen los plugins; **nunca** un `_shared` dentro de `plugins/`
  ni un import directo de `src/core` desde un plugin;
- compartido entre consumidores (chat/agent) → `uses/common/`;
- compartido en el framework agnóstico → su subsistema en `src/core/` (entrenamiento →
  `src/core/training/`, etc.).
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
su subsistema (p. ej. el daemon de consolidación en `src/core/consolidation/daemon.py`,
ejecutable con `rdmca daemon` o `python -m src.core.consolidation.daemon`). Las apps que
CONSUMEN el modelo (chat, agent) viven en `uses/`, no en `scripts/`. Al añadir un CLI
nuevo, regístralo en `COMMANDS` de `rdmca.py`.
