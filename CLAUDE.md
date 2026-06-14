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

## Arquitectura de carpetas: core · plugins · uses (OBLIGATORIO)

Tres capas, separadas a propósito:

- **`src/core/`** — el FRAMEWORK, **agnóstico al dominio**: todo lo general del modelo
  (`backend/`, `model/`, `modalities/`, `data/`, `training/`, `consolidation/`,
  `memory/`, `relevance/`, `routing/` + `config.py`, `env.py`, `resources.py`,
  `observability.py`). El objetivo es que, cambiando los plugins, este mismo framework
  pueda entrenar OTRO tipo de modelo (p. ej. reconocimiento de manos para VR). **El core
  NUNCA importa un stage concreto** (solo descubre los plugins vía el registry) ni
  importa de `uses/`.
- **`src/plugins/`** — los plugins, organizados por **DOMINIO**, **100% aislados**: un
  plugin solo se CONSUME, no depende de nada del framework salvo del **SDK de plugins**
  ([src/plugins/sdk/](src/plugins/sdk/)) — un único import estable. NUNCA importa de
  `src/core` directamente, ni de otros stages. Así borrar un plugin (`rm -rf` su
  carpeta) jamás rompe el framework ni otro stage, y añadir uno solo requiere el SDK.
  El propio sistema de plugins (`base.py`, `registry.py`, `sdk/`) vive aquí pero NO es
  un plugin.
- **`uses/`** — los CONSUMIDORES del modelo (chat, agent, futura API). NO son parte del
  framework: son formas de consumir el modelo ya creado. Lo compartido entre
  consumidores va en `uses/common/` (`agent.py`, `generate.py`, `loading.py`,
  `interaction.py`). `uses/` puede importar de `src/core` y `src/plugins`; el framework
  NUNCA importa de `uses/`.

## Dominios: un framework, varios escenarios

Cada **dominio** es un escenario de entrenamiento bajo `src/plugins/<dominio>/` y agrupa
sus propios stages:
- **`src/plugins/cognition/`** — el LLM conversacional/agéntico (los 10 stages actuales).
  Es el dominio **por defecto** (`cfg["domain"]` ausente → `cognition`).
- **`src/plugins/hands_recognition/`** — TODO (pose de manos para VR), el segundo dominio
  que demuestra que el framework es agnóstico: solo es un stub con instrucciones.

El dominio activo se elige con `cfg["domain"]` (clave del YAML del nivel). `scripts/train.py`
y `scripts/prepare_data.py` llaman `set_domain(cfg.get("domain"))` al arrancar, ANTES de
tocar el registry, para que descubra los stages de ESE dominio.

**`DomainSpec`** ([src/plugins/base.py](src/plugins/base.py)) es la costura que hace el
motor agnóstico a la tarea/modalidad. El trainer NUNCA construye el modelo, el loader, la
pérdida ni el gate directamente: se los pide al `DomainSpec` activo (resuelto en
[src/core/training/domain.py](src/core/training/domain.py)). El spec por defecto (`text-lm`)
cablea las piezas del LLM de texto (`setup.build_stage_model`, `dataload.build_data_loader`,
la objetivo MRL+aux, `gates.evaluate_gate`), así `cognition` se comporta idéntico que antes.
Un dominio sobreescribe cualquier pieza exponiendo `DOMAIN = DomainSpec(...)` (o
`build_domain(cfg) -> DomainSpec`) en su paquete. Convención: en `evaluate`, **menor score
es mejor** (una métrica higher-is-better devuelve p. ej. `1-accuracy`), para que el ratchet
del trainer siga siendo agnóstico a la métrica.

## Estructura: stages como plugins

Cada stage del currículo es un **plugin autónomo** bajo
`src/plugins/<dominio>/stageNN_<slug>/`:
- `plugin.py` — metadata del stage (número, nombre, gate, rehearsal, lr_scale,
  `trains_mood`, freeze-point, `enabled`) en un `StagePlugin`.
- `sources.py` — los generadores de datos PROPIOS del stage (un dict `SOURCES`:
  clave → builder).
- `data/level{L}/` — el corpus generado del stage vive DENTRO de su paquete
  (gitignored). Lo resuelve `src.plugins.stage_data_dir`.

El **registry** ([src/plugins/registry.py](src/plugins/registry.py)) los descubre solos
(escanea `stageNN_*` del dominio activo), valida y responde todo lo de stages (`get_stage`,
`active_stages`, `bcf_stage`, `is_behavioral`, `mood_stages`, `stream_source`,
`stage_data_dir`). `bcf_stage()` devuelve `int | None` (un dominio puede NO tener punto de
congelado → freeze opcional). Un stage es autónomo: su ÚNICO acoplamiento al framework es
`from src.plugins.sdk import ...` (contrato + helpers: `StagePlugin`, `StageGate`,
`blend`, `interleave`, `cycle_records`, `stable_hash`, `passes_filter`, `persona_for`,
`prepend_system`, `hermes_events`, `emotion_to_mood`, …). Helpers usados solo por UN
stage viven dentro de su propio paquete (p. ej.
`src/plugins/cognition/stage01_language/dictionary.py`). El SDK es el puente al core; el
plugin nunca pasa de él.

**Para añadir/quitar un stage:** suelta (o borra) un paquete
`src/plugins/<dominio>/stage11_*/`; nada más se edita. Para desactivar uno: `enabled=False`
en su plugin **o** `curriculum.stageN.enabled: false` en el YAML del nivel (lo respetan
train y prepare).

**Base a congelar:** cada stage DECLARA explícitamente si pertenece a la base
cognitiva que se congela, con `frozen_base: bool` en su `plugin.py` (True = cognitivo /
dentro de la base; False = behavioral / entrena un sector LoRA sobre el core ya
congelado). NO se decide por número de stage ni umbral — lo dice el propio stage.

**Tests por plugin:** cada stage mantiene SUS tests en
`src/plugins/<dominio>/stageNN_<slug>/tests/` (borrar el stage se lleva sus tests). Los
tests transversales (registry, pipeline, entrenamiento, y los del framework/SDK como
loader y `blend`) viven en `tests/` y NO importan ningún stage concreto. `pytest.ini`
colecciona ambos (`testpaths = tests src/plugins`); el `conftest.py` de la raíz pone el
repo en `sys.path`.

Otros puntos clave del refactor:
- Entrenador descompuesto en [src/core/training/](src/core/training/) (`trainer`,
  `setup`, `gates`, `checkpoint`, `dataload`, `valdata`, `heads`, `curriculum`); CLI en
  [scripts/train.py](scripts/train.py). El daemon en
  `python -m src.core.consolidation.daemon` (src/core/consolidation/daemon.py).
- Núcleo de generación en [uses/common/generate.py](uses/common/generate.py) y carga de
  modelo/checkpoint en [uses/common/loading.py](uses/common/loading.py) — son
  CONSUMIDORES (los reusan chat y agent), por eso viven en `uses/`, no en el framework.
- `src/core/training/stages.py` queda como **shim deprecado** que re-exporta desde las
  nuevas ubicaciones — no añadas código nuevo ahí. (El antiguo `src/data/graded.py` se
  ELIMINÓ: acoplaba el core a todos los stages e impedía borrar uno.)

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
  ni un import directo de `src/core` desde un plugin;
- compartido entre consumidores (chat/agent) → `uses/common/`;
- compartido en el framework agnóstico → su subsistema en `src/core/` (entrenamiento →
  `src/core/training/`, etc.).
Nunca copies un helper en dos sitios — muévelo arriba y que ambos lo importen.

**Scripts vs internos.** En [scripts/](scripts/) van SOLO los CLIs accesibles para el
developer (train, prepare_data, train_tokenizer, run_benchmarks, plot_metrics, purge,
ood_probe, prepare_multimodal). Los componentes de runtime/internos NO van en scripts:
viven en su subsistema (p. ej. el daemon de consolidación en
`src/core/consolidation/daemon.py`, ejecutable con `python -m src.core.consolidation.daemon`).
Las apps que CONSUMEN el modelo (chat, agent) viven en `uses/`, no en `scripts/`.
