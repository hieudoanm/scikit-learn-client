# skl — ONNX

- Convert the **full sklearn pipeline** with `skl2onnx`.
- Pin `target_opset` (default 15) via env `ONNX_OPSET`.
- Verify with `onnx.checker.check_model` before insert.
- CI: `onnxruntime` InferenceSession, one `run` with dummy floats matching `n_features` after transform **or** use `zipmap=False` so outputs are tensors.
- Optional pickle uses `joblib.dump` of the same pipeline; never a substitute for ONNX. A `.joblib` artefact is the joblib serialisation of the pipeline and is a separate optional format from `.pkl`.
