"""Model: **hands_recognition** — TODO (hand-pose recognition for VR).

A second training scenario, to prove the framework is model-agnostic: swap the model
(`cfg["model_name"] = "hands_recognition"`) and the same engine trains a different kind
of model. Nothing is implemented yet — this package only marks where the model goes.

To build it (see ModelSpec in src/models/base.py and the cognition model for the
reference wiring):
  1. Add stage plugins here, e.g. `stage01_keypoints/`, each exposing a `PLUGIN`
     (StagePlugin) — the registry discovers them when this model is active.
  2. Provide a `SPEC = ModelSpec(...)` that supplies the non-text pieces:
        • build_model     — a hand-pose encoder/head (not the text transformer)
        • build_loader    — an image/pose dataset loader (not TextDataset)
        • objective       — keypoint regression / classification loss
        • gate_metric     — e.g. "pck" / "accuracy" (higher-is-better), evaluated by a
                            gate evaluator registered for that metric_key.
  3. Write a level config with `model_name: hands_recognition`.
  4. Put its hypothesis probes under `experiments/` within this package.
"""
