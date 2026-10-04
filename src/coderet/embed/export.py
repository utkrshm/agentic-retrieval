"""Export jina-code to OpenVINO, optionally compress weights to int8, and store probe references.

Layout under ``<root>/<key>-<revision8>/``::

    fp32/openvino_model.{xml,bin}     exported graph (always created)
    int8/openvino_model.{xml,bin}     int8 weight-compressed copy (NNCF), if requested
    probe_reference.npy               fp32 PyTorch vectors for the health probe items
    meta.json                         what was exported, with library versions

The reference vectors let the encoder reject an export that loads but computes the wrong thing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from coderet.config import ModelSpec
from coderet.embed.probe import PROBE_ITEMS, save_reference

DEFAULT_ROOT = Path(".cache/openvino")


def export_dir(spec: ModelSpec, root: Path = DEFAULT_ROOT) -> Path:
    return Path(root) / f"{spec.key}-{spec.revision[:8]}"


def export_openvino(spec: ModelSpec, root: Path = DEFAULT_ROOT, int8: bool = True,
                    force: bool = False) -> Path:
    """Create the exported model(s) and probe reference. Returns the export directory."""
    import nncf
    import openvino as ov
    from optimum.intel import OVModelForFeatureExtraction

    out = export_dir(spec, root)
    fp32_xml = out / "fp32" / "openvino_model.xml"
    if force or not fp32_xml.exists():
        t = time.perf_counter()
        model = OVModelForFeatureExtraction.from_pretrained(
            spec.hf_id, revision=spec.revision, export=True, trust_remote_code=True, compile=False
        )
        model.save_pretrained(str(out / "fp32"))
        del model
        print(f"exported fp32 graph in {time.perf_counter() - t:.0f}s")

    if int8:
        int8_xml = out / "int8" / "openvino_model.xml"
        if force or not int8_xml.exists():
            core = ov.Core()
            compressed = nncf.compress_weights(core.read_model(str(fp32_xml)),
                                               mode=nncf.CompressWeightsMode.INT8_ASYM)
            (out / "int8").mkdir(parents=True, exist_ok=True)
            ov.save_model(compressed, str(int8_xml))
            print("compressed weights to int8")

    # reference vectors from the fp32 PyTorch backend (the thing we trust)
    from coderet.embed.backends import TorchBackend

    ref_path = out / "probe_reference.npy"
    if force or not ref_path.exists():
        import numpy as np

        torch_backend = TorchBackend(spec, "cpu")
        rows = {}
        for role in ("query", "document"):
            idx = [i for i, (r, _) in enumerate(PROBE_ITEMS) if r == role]
            vecs = torch_backend.embed([PROBE_ITEMS[i][1] for i in idx], role)
            rows.update(zip(idx, vecs, strict=True))
        save_reference(ref_path, np.vstack([rows[i] for i in range(len(PROBE_ITEMS))]))

    import importlib.metadata as md

    (out / "meta.json").write_text(json.dumps({
        "model": spec.hf_id, "revision": spec.revision, "int8": int8,
        "openvino": md.version("openvino"), "nncf": md.version("nncf"),
        "optimum-intel": md.version("optimum-intel"), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    return out
