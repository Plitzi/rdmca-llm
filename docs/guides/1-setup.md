← [README](../../README.md) | → [2-data.md](2-data.md)

# Setup inicial

## Requisitos

- MacBook con Apple Silicon (M1 / M2 / M3 / M4)
- macOS con Homebrew instalado
- ~40 GB libres en disco (para datos + pesos + venv)

---

## Instalación (una sola vez)

```bash
# 1. Entrar al directorio del proyecto
cd rdmca-llm

# 2. Crear entorno virtual con Python 3.10 de Homebrew
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install mlx mlx-lm sentencepiece pyyaml numpy tqdm datasets pytest rich

# 4. Verificar que MLX usa el GPU de Apple Silicon
python -c "import mlx.core as mx; print(mx.default_device())"
# Esperado: Device(gpu, 0)
# Si dice "cpu": reinstalar mlx con pip install --upgrade mlx
```

---

## Activar el entorno en sesiones futuras

```bash
source .venv/bin/activate
```

> Los scripts principales (`train_stage.py`, `chat.py`, `consolidation_daemon.py`)
> se auto-reinician con el Python del venv si los corrés sin activar el entorno.
> El activado manual es solo necesario para correr `pytest` o los scripts de `scripts/`.

---

## Dependencias y para qué sirven

| Paquete | Uso |
|---|---|
| `mlx` | Framework de ML nativo Apple Silicon (GPU unificado) |
| `mlx-lm` | Utilidades de lenguaje sobre MLX |
| `sentencepiece` | Tokenizador BPE bilingüe |
| `pyyaml` | Leer configs `.yaml` |
| `numpy` | Operaciones numéricas auxiliares |
| `tqdm` | Barras de progreso |
| `datasets` | Descarga de corpora desde HuggingFace |
| `pytest` | Tests unitarios de los módulos |
